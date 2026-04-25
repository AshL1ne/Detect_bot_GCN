import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from collections import Counter


# ==================== 配置 ====================
GRAPH_PATH = "../data_process/graph_data.pt"
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

# ---------- 对未知用户进行预测 ----------
unknown_mask = (data.y == -1)
if unknown_mask.sum() > 0:
    print(f"\n对 {unknown_mask.sum().item()} 个未知用户进行预测...")
    model.eval()
    # 用torch.no_grad()关闭梯度计算，彻底解决报错
    with torch.no_grad():
        out = model(data)
        prob = out.exp()  # 转换为类别概率
        mal_prob = prob[unknown_mask][:, 1].cpu().numpy()  # 提取恶意用户概率
    # 分类阈值，可根据验证集效果调整
    threshold = 0.5
    pred_labels = (mal_prob >= threshold).astype(int)

    print("恶意概率均值: {:.4f}，中位数: {:.4f}".format(mal_prob.mean(), np.median(mal_prob)))
    print(f"预测高风险恶意用户数量: {(pred_labels == 1).sum()}")

    # 【毕设加分可选】把预测结果和用户ID关联，保存为CSV文件
    # 先在data_preprocess.py末尾加上这行，保存用户ID列表：
    #
    # 取消下方注释即可生成结果文件
    import pandas as pd
    user_id_df = pd.read_csv('../data_process/user_id_list.csv')
    unknown_user_ids = user_id_df[unknown_mask.cpu().numpy()]['user_id'].values
    result_df = pd.DataFrame({
        'user_id': unknown_user_ids,
        'malicious_prob': mal_prob,
        'pred_label': pred_labels
    })
    result_df.to_csv('malicious_user_predictions.csv', index=False, encoding='utf-8-sig')
    print("预测结果已保存至 malicious_user_predictions.csv")

else:
    print("没有未知用户需要预测。")