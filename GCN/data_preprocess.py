import os, json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DIR = "../data"  # JSONL 文件夹路径
OUT_GRAPH = "../output/graph_data.pt"           # 输出的图数据文件
RANDOM_SEED = 42
# ==============================================

# ---------- 1. 读取 JSONL ----------
def load_jsonl_files(dir_path, prefix):
    records = []
    for fname in os.listdir(dir_path):
        if fname.startswith(prefix) and fname.endswith('.jsonl'):
            with open(os.path.join(dir_path, fname), 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
    return records

print("正在加载数据...")
user_records = load_jsonl_files(DIR, 'user')
fan_records = load_jsonl_files(DIR, 'fan')
fan_fan_records = load_jsonl_files(DIR, 'fan_fan')
follow_records = load_jsonl_files(DIR, 'follow')
tweet_records = load_jsonl_files(DIR, 'tweet')

# 合并粉丝关系（fan + fan_fan）
all_fan_records = fan_records + fan_fan_records

tweets_df = pd.DataFrame(tweet_records)
print(f"用户: {len(user_records)}, 关注: {len(follow_records)}, 粉丝关系: {len(all_fan_records)}, 微博: {len(tweets_df)}")

# ---------- 2. 合并所有用户节点 ----------
all_user_dict = {}
for rec in user_records:
    all_user_dict[rec['_id']] = rec
# 从关系数据中提取嵌套用户
for rec in all_fan_records:
    info = rec.get('fan_info', {})
    if info and info['_id'] not in all_user_dict:
        all_user_dict[info['_id']] = info
for rec in follow_records:
    info = rec.get('follow_info', {})
    if info and info['_id'] not in all_user_dict:
        all_user_dict[info['_id']] = info

full_users = pd.DataFrame(all_user_dict.values())
full_users['_id'] = full_users['_id'].astype(str)
print(f"图中总用户数: {len(full_users)}")

# ---------- 3. 基础特征提取 ----------
# 数值特征
for col in ['followers_count', 'follow_count', 'statuses_count', 'mbrank', 'mbtype']:
    full_users[col] = pd.to_numeric(full_users[col], errors='coerce').fillna(0)
full_users['verified'] = full_users['verified'].fillna(False).astype(int)
full_users['gender'] = full_users['gender'].map({'m': 0, 'f': 1, '': -1}).fillna(-1).astype(int)

# ========== 新增特征：关注/粉丝比 ==========
# +1 防止除零，该比值越大表示用户更偏向“关注者”而非被关注者
full_users['follower_follow_ratio'] = full_users['follow_count'] / (full_users['followers_count'] + 1)
# ============================================

# 关键词检测函数
WX_WORDS = ['微信', '微x','v信','vx', 'VX', 'wechat', 'WeChat', 'wx']
INVEST_WORDS = ['投资', '月入' '理财', '分红']
FRIEND_WORDS = ['交友', '真诚交友', '诚心交友']

def contains_any(text, words):
    if pd.isna(text):
        return False
    return any(w in str(text) for w in words)

full_users['desc_has_wx'] = full_users['description'].apply(lambda x: contains_any(x, WX_WORDS)).astype(int)
full_users['desc_has_invest'] = full_users['description'].apply(lambda x: contains_any(x, INVEST_WORDS)).astype(int)
full_users['nick_has_wx'] = full_users['nick_name'].apply(lambda x: contains_any(x, WX_WORDS)).astype(int)

# ---------- 4. 微博行为特征 ----------
# 预处理微博表
tweets_df['user_id'] = tweets_df['user_id'].astype(str)
# 解析时间（格式: "Wed Apr 01 22:20:19 +0800 2026"）
tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'],
                                         format='%a %b %d %H:%M:%S %z %Y',
                                         errors='coerce')
tweets_df['is_forward'] = tweets_df['is_retweet'].fillna(False).astype(bool)

# 安全转发关键词
SAFE_FORWARD_KW = ['转评赞', '抽取', '抽送','随机抽', '锦鲤', '好运']

def is_safe_forward(row):
    """判断转发是否属于安全转发：被转发的原微博内容包含安全关键词"""
    if not row['is_forward']:
        return False
    retweet_info = row.get('retweet_info', {})
    if not isinstance(retweet_info, dict):
        return False
    ret_content = retweet_info.get('content', '')
    if pd.isna(ret_content):
        return False
    return contains_any(ret_content, SAFE_FORWARD_KW)

tweets_df['safe_forward'] = tweets_df.apply(is_safe_forward, axis=1)

# 按用户聚合微博特征
user_tweet_feats = defaultdict(lambda: {
    'original_count': 0,
    'forward_count': 0,
    'total_tweets': 0,
    'original_ratio': 0.0,
    'avg_interact': 0.0,
    'freq_forward_bursts': 0,
    'friendword_ratio': 0.0
})

for uid, grp in tqdm(tweets_df.groupby('user_id'), desc="聚合微博特征"):
    feats = user_tweet_feats[uid]
    total = len(grp)
    feats['total_tweets'] = total
    originals = (~grp['is_forward']).sum()
    forwards = grp['is_forward'].sum()
    feats['original_count'] = originals
    feats['forward_count'] = forwards

    # 原创微博占比 = 原创数 / 总微博数，避免除零
    feats['original_ratio'] = originals / total if total > 0 else 0.0

    # 平均互动
    interact = grp['reposts_count'].fillna(0) + grp['attitudes_count'].fillna(0)
    feats['avg_interact'] = interact.mean() if total > 0 else 0.0

    # 频繁转发（排除安全转发）
    fwd_df = grp[grp['is_forward'] & (~grp['safe_forward'])].sort_values('created_at')
    burst_count = 0
    if len(fwd_df) > 1:
        time_diff = fwd_df['created_at'].diff().dt.total_seconds().dropna()
        burst_count = (time_diff < 600).sum()   # 10分钟 = 600秒
    feats['freq_forward_bursts'] = burst_count

    # 交友内容比例（检测用户自己发布的内容，含转发评语）
    friend_cnt = grp['content'].apply(lambda x: contains_any(x, FRIEND_WORDS)).sum()
    feats['friendword_ratio'] = friend_cnt / total if total > 0 else 0.0

# 将微博特征合并到用户表
tweet_feat_df = pd.DataFrame.from_dict(user_tweet_feats, orient='index')
tweet_feat_df.index.name = '_id'
full_users = full_users.join(tweet_feat_df, on='_id')
# 无微博的用户填充0
for c in tweet_feat_df.columns:
    full_users[c] = full_users[c].fillna(0)

# ---------- 5. 特征矩阵 ----------
feature_cols = [
    'followers_count', 'follow_count', 'statuses_count',
    'mbrank', 'mbtype', 'verified', 'gender',
    'desc_has_wx', 'desc_has_invest', 'nick_has_wx',
    'original_count', 'forward_count', 'total_tweets','original_ratio',
    'avg_interact', 'freq_forward_bursts', 'friendword_ratio',
    'follower_follow_ratio'
]
# 确保没有缺失
full_users[feature_cols] = full_users[feature_cols].fillna(0).astype(float)

X = full_users[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_tensor = torch.tensor(X_scaled, dtype=torch.float)

# ---------- 6. 标签 ----------
label_map = {'normal': 0, 'malicious': 1, 'unknown': -1}
y = full_users['user_type'].map(label_map).fillna(-1).astype(int).tolist()
y_tensor = torch.tensor(y, dtype=torch.long)

# ---------- 7. 构建图 ----------
# 节点ID映射
user_id_list = full_users['_id'].tolist()
user_to_idx = {uid: i for i, uid in enumerate(user_id_list)}

edges = set()
def add_edge(src, dst):
    if src in user_to_idx and dst in user_to_idx:
        edges.add((user_to_idx[src], user_to_idx[dst]))

# 关注关系
for rec in follow_records:
    fid = str(rec.get('fan_id'))
    tid = str(rec.get('follow_id'))
    add_edge(fid, tid)
# 粉丝关系（fan和fan_fan）
for rec in all_fan_records:
    fan_id = str(rec.get('fan_id'))
    followed_id = str(rec.get('followed_id'))
    add_edge(fan_id, followed_id)

# 转为无向边（双向对称）
undirected = set()
for s, t in edges:
    undirected.add((s, t))
    undirected.add((t, s))
edge_list = list(undirected)
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

print(f"图节点: {x_tensor.shape[0]}, 边数: {edge_index.shape[1]}")

# ---------- 8. 划分训练/验证/测试掩码 ----------
from sklearn.model_selection import train_test_split

labeled_mask = y_tensor >= 0
labeled_idx = np.where(labeled_mask)[0]
labels_labeled = y_tensor[labeled_idx].numpy()

idx_train, idx_tmp, _, y_tmp = train_test_split(
    labeled_idx, labels_labeled, test_size=0.4, stratify=labels_labeled, random_state=RANDOM_SEED
)
idx_val, idx_test = train_test_split(
    idx_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_SEED
)

train_mask = torch.zeros(len(full_users), dtype=torch.bool)
val_mask = torch.zeros(len(full_users), dtype=torch.bool)
test_mask = torch.zeros(len(full_users), dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

print(f"训练集: {train_mask.sum().item()}, 验证集: {val_mask.sum().item()}, 测试集: {test_mask.sum().item()}")

# ---------- 9. 封装为 Data 对象并保存 ----------
data = Data(
    x=x_tensor,
    edge_index=edge_index,
    y=y_tensor,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    num_classes=2
)
torch.save(data, OUT_GRAPH)
print(f"图数据已保存至 {OUT_GRAPH}")

# ===================== 新增：导出 CSV =====================
# 1. 用户原始表（不含模型预测，只保留后端需要的列）
user_out_cols = [
    '_id', 'nick_name', 'description', 'gender',
    'followers_count', 'follow_count', 'statuses_count',
    'verified', 'mbrank', 'mbtype', 'user_type',   # 保留原始标签以便核对
    'original_count', 'forward_count'
]
users_raw_df = full_users[user_out_cols].copy()
users_raw_df.to_csv("../output/users_raw.csv", index=False, encoding='utf-8-sig')
print("用户原始信息已保存至 ../output/users_raw.csv")

# 2. 关系表（关注关系，不分来源）
rel_rows = []
for rec in follow_records:
    follower = str(rec.get('fan_id'))
    followee = str(rec.get('follow_id'))
    if follower in user_to_idx and followee in user_to_idx:
        rel_rows.append({'follower_id': follower, 'followee_id': followee})

for rec in all_fan_records:
    follower = str(rec.get('fan_id'))
    followee = str(rec.get('followed_id'))
    if follower in user_to_idx and followee in user_to_idx:
        rel_rows.append({'follower_id': follower, 'followee_id': followee})

relations_df = pd.DataFrame(rel_rows).drop_duplicates()
relations_df.to_csv("../output/relations.csv", index=False, encoding='utf-8-sig')
print(f"关系表已保存至 ../output/relations.csv，共 {len(relations_df)} 条关系")