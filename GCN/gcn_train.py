import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from collections import Counter
import pandas as pd


# ==================== 配置 ====================
GRAPH_PATH = "../output/graph_data.pt"
EPOCHS = 300
LR = 0.01
HIDDEN = 64
DROP_OUT = 0.5
WEIGHT_DECAY = 5e-4
RANDOM_SEED = 42
# ==============================================

torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载图数据
data = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
print(f"加载图数据：节点数 {data.num_nodes}, 特征维度 {data.num_features}")

# ---------- GCN 模型 ----------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(
    in_dim=data.num_features,
    hidden_dim=HIDDEN,
    out_dim=data.num_classes,
    dropout=DROP_OUT
).to(device)
data = data.to(device)

# ---------- 损失函数（加权） ----------
# 计算类别权重缓解不平衡
labels = data.y[data.train_mask].cpu().numpy()
cnt = Counter(labels)
# 权重 = 总样本数 / (类别数 * 该类样本数)
total = sum(cnt.values())
weights = [total / (2 * cnt.get(cls, 1)) for cls in range(2)]
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = torch.nn.NLLLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ---------- 训练 & 评估 ----------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    acc = correct.item() / mask.sum().item()
    # 计算 AUC（仅对二分类）
    prob = out.exp()[:, 1]   # 恶意类概率
    true = data.y[mask].cpu().numpy()
    score = prob[mask].cpu().numpy()
    auc = roc_auc_score(true, score) if len(np.unique(true)) > 1 else 0.5
    return acc, auc, pred, out

best_val_acc = 0
best_model_state = None

print("开始训练...")
for epoch in range(1, EPOCHS + 1):
    loss = train()
    if epoch % 20 == 0 or epoch == 1:
        val_acc, val_auc, _, _ = evaluate(data.val_mask)
        test_acc, test_auc, _, _ = evaluate(data.test_mask)
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val Acc {val_acc:.4f} AUC {val_auc:.4f} | Test Acc {test_acc:.4f} AUC {test_auc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

# 加载最佳模型
model.load_state_dict(best_model_state)
print("\n最佳验证集准确率: {:.4f}".format(best_val_acc))

# ---------- 最终测试评估 ----------
print("\n=== 测试集评估 ===")
test_acc, test_auc, test_pred, test_out = evaluate(data.test_mask)
print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

# 分类报告
test_true = data.y[data.test_mask].cpu().numpy()
test_pred = test_pred[data.test_mask].cpu().numpy()
print(classification_report(test_true, test_pred, target_names=['normal','malicious'], digits=4))

# ---------- 对未知用户进行预测（保持原有逻辑，同时计算所有节点的恶意概率）----------
unknown_mask = (data.y == -1)
if unknown_mask.sum() > 0:
    print(f"\n对 {unknown_mask.sum().item()} 个未知用户进行预测...")

# 对所有节点做一次推理，得到恶意概率
model.eval()
with torch.no_grad():
    logits = model(data)
    prob = logits.exp()       # shape: [num_nodes, 2]
    mal_prob_all = prob[:, 1].cpu().numpy()    # 第二个类（恶意）的概率
    pred_all = (mal_prob_all >= 0.5).astype(int)

# 输出未知用户统计
if unknown_mask.sum() > 0:
    mal_prob_unknown = mal_prob_all[unknown_mask.cpu().numpy()]
    pred_unknown = pred_all[unknown_mask.cpu().numpy()]
    print("恶意概率均值: {:.4f}，中位数: {:.4f}".format(mal_prob_unknown.mean(), np.median(mal_prob_unknown)))
    print(f"预测高风险恶意用户数量: {(pred_unknown == 1).sum()}")

    # 保存训练好的模型（state_dict）
    torch.save(model.state_dict(), "../output/GCN_model.pt")
    print("模型已保存至 ../output/GCN_model.pt")

    # --------------------- 合并用户信息并导出最终表 ---------------------
    print("\n正在生成最终用户表...")
    users_raw = pd.read_csv("../output/users_raw.csv", dtype={'_id': str})
    # 确保顺序与图节点一致（data_preprocess中按user_id_list顺序构造特征，users_raw正是以此顺序保存）
    # 添加模型预测结果
    users_raw['mal_prob'] = mal_prob_all
    users_raw['is_malicious'] = pred_all.astype(int)

    # 可选：将预测标签转换为更易读的文本
    # users_raw['predicted_type'] = users_raw['is_malicious'].map({0: 'normal', 1: 'malicious'})

    # 保存最终用户表（包含原始 user_type 以便对比，最终交付可移除该列）
    users_raw.drop(columns=['user_type'], inplace=True)
    users_raw.to_csv("../output/users.csv", index=False, encoding='utf-8-sig')
    print("最终用户表已保存至 ../output/users.csv")
    print("包含列：", users_raw.columns.tolist())