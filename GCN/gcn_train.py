import os
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)

# ==================== 配置 ====================
GRAPH_PATH = "../output/graph_data.pt"
EPOCHS = 300
LR = 0.01
HIDDEN = 64
DROP_OUT = 0.5
WEIGHT_DECAY = 5e-4
RANDOM_SEED = 42

MODEL_OUT = "../output/GCN_model.pt"
PLOT_DIR = "../output/plots"
PRINT_EVERY = 20

LOSS_PNG = "loss_curve.png"
ACC_PNG = "acc_curve.png"
AUC_PNG = "auc_curve.png"
PREC_PNG = "precision_curve.png"
REC_PNG = "recall_curve.png"
F1_PNG = "f1_curve.png"

# 网络图输出
NET_DIR = os.path.join(PLOT_DIR, "networks")

# 网络图参数
LAYOUT_SEED = 42
NODE_ALPHA = 0.95
EDGE_COLOR = "#7f7f7f"

# 全图
FULL_LAYOUT_ITERS = 200
FULL_NODE_SIZE = 3
FULL_EDGE_ALPHA = 0.15
FULL_EDGE_WIDTH = 0.5

# 同类诱导子图
SUB_LAYOUT_ITERS = 250
SUB_NODE_SIZE = 5
SUB_EDGE_ALPHA = 0.25
SUB_EDGE_WIDTH = 0.5

# 2-hop 子图
HOP_LAYOUT_ITERS = 220
HOP_NODE_SIZE = 5
HOP_EDGE_ALPHA = 0.20
HOP_EDGE_WIDTH = 0.5
HOPS = 2

# 2-hop：中心选择策略
MAL_TOPK = 50          # 恶意中心 Top-K（按 mal_prob_all 从高到低）
DRAW_ALL_CENTERS_2HOP = True  # 是否仍输出“全部中心”的 2-hop（用于对比）

# 只画最大连通分量（避免碎片太多）
KEEP_LARGEST_CC = True

# 颜色：normal 蓝，malicious 红
COLOR_NORMAL = "#1f77b4"
COLOR_MAL = "#d62728"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# 固定随机种子，保证实验可复现
torch.manual_seed(RANDOM_SEED) # PyTorch 的随机数
np.random.seed(RANDOM_SEED)    # NumPy 的随机数
random.seed(RANDOM_SEED)       # Python 自带的随机数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载图数据
data = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
print(f"加载图数据：节点数 {data.num_nodes}, 特征维度 {data.num_features}")
data = data.to(device)


# GCN
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)   # 输入特征 -> 隐藏层 (64维)
        x = F.relu(x)                  # ReLU激活
        x = F.dropout(x, p=self.dropout, training=self.training) # Dropout-0.5，防止过拟合
        x = self.conv2(x, edge_index)  # 隐藏层 -> 输出层 (2维，二分类)
        return F.log_softmax(x, dim=1) # log_softmax用于分类


model = GCN(
    in_dim=data.num_features,
    hidden_dim=HIDDEN,
    out_dim=data.num_classes,
    dropout=DROP_OUT,
).to(device)


# 损失函数（加权
labels = data.y[data.train_mask].detach().cpu().numpy()
cnt = Counter(labels) # 如{0: 300, 1: 100}
total = sum(cnt.values())
# 权重 = 总样本数 ÷ (类别数 × 当前类别样本数)
weights = [total / (2 * cnt.get(cls, 1)) for cls in range(2)]
class_weights = torch.tensor(weights, dtype=torch.float, device=device)
criterion = torch.nn.NLLLoss(weight=class_weights)  # 带权重的负对数似然损失
# Adam 的优化器，根据损失自动修改模型的权重
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# 指标
def binary_prf(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0
    # 不需要返回support
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
    )
    return float(p), float(r), float(f1)


@torch.no_grad()
def eval_acc_auc(mask):
    model.eval() # 评估模式（关闭 dropout 等训练专用机制）
    out = model(data)
    pred = out.argmax(dim=1)

    correct = (pred[mask] == data.y[mask]).sum()
    acc = correct.item() / mask.sum().item()

    # exp变回正常概率
    prob_mal = out.exp()[:, 1]
    true = data.y[mask].detach().cpu().numpy()
    score = prob_mal[mask].detach().cpu().numpy()
    auc = roc_auc_score(true, score) if len(np.unique(true)) > 1 else 0.5
    return acc, auc


@torch.no_grad()
def eval_test_full():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    mask = data.test_mask
    correct = (pred[mask] == data.y[mask]).sum()
    acc = correct.item() / mask.sum().item()

    prob_mal = out.exp()[:, 1]
    true = data.y[mask].detach().cpu().numpy()
    score = prob_mal[mask].detach().cpu().numpy()
    auc = roc_auc_score(true, score) if len(np.unique(true)) > 1 else 0.5

    y_pred = pred[mask].detach().cpu().numpy()
    p, r, f1 = binary_prf(true, y_pred, pos_label=1)

    return {
        "acc": acc,
        "auc": auc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "pred_all": pred,
        "out_all": out,
    }


# 训练
def train_one_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward() # 反向传播算梯度
    optimizer.step() # 改权重
    return float(loss.item())


history = {
    "epoch": [],
    "train_loss": [],
    "test_acc": [],
    "test_auc": [],
    "test_precision": [],
    "test_recall": [],
    "test_f1": [],
}

best_val_acc = -1.0
best_state = None

print("开始训练...")
for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch()

    val_acc, val_auc = eval_acc_auc(data.val_mask)
    test = eval_test_full()

    history["epoch"].append(epoch)
    history["train_loss"].append(loss)
    history["test_acc"].append(test["acc"])
    history["test_auc"].append(test["auc"])
    history["test_precision"].append(test["precision"])
    history["test_recall"].append(test["recall"])
    history["test_f1"].append(test["f1"])

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | Loss {loss:.4f} | "
            f"Val Acc {val_acc:.4f} AUC {val_auc:.4f} | "
            f"Test Acc {test['acc']:.4f} AUC {test['auc']:.4f} "
            f"P {test['precision']:.4f} R {test['recall']:.4f} F1 {test['f1']:.4f}"
        )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)

print(f"\n最佳验证集准确率: {best_val_acc:.4f}")


# 最终测试评估
print("\n=== 测试集评估 ===")
test = eval_test_full()
print(
    f"Test Accuracy: {test['acc']:.4f}, Test AUC: {test['auc']:.4f}, "
    f"Test Precision: {test['precision']:.4f}, Test Recall: {test['recall']:.4f}, Test F1: {test['f1']:.4f}"
)

test_true = data.y[data.test_mask].detach().cpu().numpy()
test_pred = test["pred_all"][data.test_mask].detach().cpu().numpy()
print(classification_report(test_true, test_pred, target_names=["normal", "malicious"], digits=4))


# 推理所有节点（用于画图 & 导出表）+ 保存模型
unknown_mask = (data.y == -1)
if unknown_mask.sum() > 0:
    print(f"\n对 {unknown_mask.sum().item()} 个未知用户进行预测...")

model.eval()
with torch.no_grad():
    logits = model(data)
    prob = logits.exp()
    mal_prob_all = prob[:, 1].detach().cpu().numpy()
    pred_all = (mal_prob_all >= 0.5).astype(int)

if unknown_mask.sum() > 0:
    mal_prob_unknown = mal_prob_all[unknown_mask.detach().cpu().numpy()]
    pred_unknown = pred_all[unknown_mask.detach().cpu().numpy()]
    print("恶意概率均值: {:.4f}，中位数: {:.4f}".format(mal_prob_unknown.mean(), np.median(mal_prob_unknown)))
    print(f"预测高风险恶意用户数量: {(pred_unknown == 1).sum()}")

torch.save(model.state_dict(), MODEL_OUT)
print(f"模型已保存至 {MODEL_OUT}")


# 最终用户表
print("\n正在生成最终用户表...")
users_raw = pd.read_csv("../output/users_raw.csv", dtype={"_id": str})
users_raw["mal_prob"] = mal_prob_all
users_raw["is_malicious"] = pred_all.astype(int)
users_raw.drop(columns=["user_type"], inplace=True)
users_raw.to_csv("../output/users.csv", index=False, encoding="utf-8-sig")
print("最终用户表已保存至 ../output/users.csv")
print("包含列：", users_raw.columns.tolist())

'''
# 曲线
ensure_dir(PLOT_DIR)
import matplotlib.pyplot as plt

# 中文字体（避免中文显示成方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def save_curve(y, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], y)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

save_curve(history["train_loss"], "loss", "Train Loss", os.path.join(PLOT_DIR, LOSS_PNG))
save_curve(history["test_acc"], "acc", "Test Accuracy", os.path.join(PLOT_DIR, ACC_PNG))
save_curve(history["test_auc"], "auc", "Test AUC", os.path.join(PLOT_DIR, AUC_PNG))
save_curve(history["test_precision"], "precision", "Test Precision", os.path.join(PLOT_DIR, PREC_PNG))
save_curve(history["test_recall"], "recall", "Test Recall", os.path.join(PLOT_DIR, REC_PNG))
save_curve(history["test_f1"], "f1", "Test F1-score", os.path.join(PLOT_DIR, F1_PNG))

print(f"\n曲线图已保存至 {PLOT_DIR}")


# 网络图：全图 + 同类诱导子图 + 2-hop（全部中心 & 恶意 Top-K）
def build_nx_graph(edge_index):
    import networkx as nx
    g = nx.Graph()
    ei = edge_index.detach().cpu().numpy()
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return g

def keep_largest_connected_component(g):
    import networkx as nx
    if g.number_of_nodes() == 0:
        return g
    largest = max(nx.connected_components(g), key=len)
    return g.subgraph(largest).copy()

def pred_color(v: int):
    return COLOR_MAL if int(v) == 1 else COLOR_NORMAL

def draw_full_graph_by_pred(g_all, pred_labels, out_path):
    import networkx as nx
    import matplotlib.pyplot as plt

    g = g_all
    if KEEP_LARGEST_CC:
        g = keep_largest_connected_component(g)

    node_colors = [pred_color(pred_labels[n]) for n in g.nodes()]

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(g, seed=LAYOUT_SEED, iterations=FULL_LAYOUT_ITERS)
    nx.draw_networkx_edges(g, pos, alpha=FULL_EDGE_ALPHA, width=FULL_EDGE_WIDTH, edge_color=EDGE_COLOR)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=FULL_NODE_SIZE, alpha=NODE_ALPHA)

    plt.title(f"全用户关系图 nodes={g.number_of_nodes()} edges={g.number_of_edges()}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def draw_induced_subgraph_by_pred_label(g_all, pred_labels, label_value, color, title_cn, out_path):
    import networkx as nx
    import matplotlib.pyplot as plt

    nodes = np.where(pred_labels == label_value)[0].tolist()
    sg = g_all.subgraph(nodes).copy()

    if KEEP_LARGEST_CC:
        sg = keep_largest_connected_component(sg)

    if sg.number_of_nodes() == 0:
        print(f"{title_cn}：子图为空，跳过。")
        return

    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(sg, seed=LAYOUT_SEED, iterations=SUB_LAYOUT_ITERS)

    nx.draw_networkx_edges(sg, pos, alpha=SUB_EDGE_ALPHA, width=SUB_EDGE_WIDTH, edge_color=color)
    nx.draw_networkx_nodes(sg, pos, node_size=SUB_NODE_SIZE, node_color=color, alpha=NODE_ALPHA)

    plt.title(f"{title_cn}（nodes={sg.number_of_nodes()} edges={sg.number_of_edges()}）")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def k_hop_nodes(g, centers, hops=2):
    visited = set(centers)
    frontier = set(centers)
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            nxt.update(g.neighbors(u))
        frontier = nxt - visited
        visited |= frontier
    return visited

def select_centers_all(pred_labels, centers_label_value):
    return np.where(pred_labels == centers_label_value)[0].tolist()

def select_centers_mal_topk(mal_prob, k):
    order = np.argsort(-mal_prob)  # 降序
    return order[: min(k, len(order))].tolist()

def draw_khop_subgraph(g_all, pred_labels, centers, title_cn, out_path):
    import networkx as nx
    import matplotlib.pyplot as plt

    if centers is None or len(centers) == 0:
        print(f"{title_cn}：中心集合为空，跳过。")
        return

    nodes = k_hop_nodes(g_all, centers, hops=HOPS)
    sg = g_all.subgraph(nodes).copy()

    if KEEP_LARGEST_CC:
        sg = keep_largest_connected_component(sg)

    node_colors = [pred_color(pred_labels[n]) for n in sg.nodes()]

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(sg, seed=LAYOUT_SEED, iterations=HOP_LAYOUT_ITERS)
    nx.draw_networkx_edges(sg, pos, alpha=HOP_EDGE_ALPHA, width=HOP_EDGE_WIDTH, edge_color=EDGE_COLOR)
    nx.draw_networkx_nodes(sg, pos, node_color=node_colors, node_size=HOP_NODE_SIZE, alpha=NODE_ALPHA)

    plt.title(f"{title_cn}（{HOPS}-hop） nodes={sg.number_of_nodes()} edges={sg.number_of_edges()}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

try:
    import networkx as nx  # noqa: F401

    ensure_dir(NET_DIR)
    g_all = build_nx_graph(data.edge_index)

    # 1) 全图
    out_full = os.path.join(NET_DIR, "full_social_network_by_pred.png")
    draw_full_graph_by_pred(g_all, pred_all, out_full)

    # 2) 同类诱导子图
    out_normal = os.path.join(NET_DIR, "pred_normal_social_graph.png")
    out_mal = os.path.join(NET_DIR, "pred_malicious_social_graph.png")

    draw_induced_subgraph_by_pred_label(
        g_all=g_all,
        pred_labels=pred_all,
        label_value=0,
        color=COLOR_NORMAL,
        title_cn="正常用户社交网络图（同类边）",
        out_path=out_normal,
    )

    draw_induced_subgraph_by_pred_label(
        g_all=g_all,
        pred_labels=pred_all,
        label_value=1,
        color=COLOR_MAL,
        title_cn="恶意用户社交网络图（同类边）",
        out_path=out_mal,
    )

    # 3) 2-hop（全部中心）
    if DRAW_ALL_CENTERS_2HOP:
        out_normal_2hop = os.path.join(NET_DIR, "pred_normal_centers_2hop_all.png")
        out_mal_2hop = os.path.join(NET_DIR, "pred_malicious_centers_2hop_all.png")

        normal_centers_all = select_centers_all(pred_all, 0)
        mal_centers_all = select_centers_all(pred_all, 1)

        draw_khop_subgraph(
            g_all=g_all,
            pred_labels=pred_all,
            centers=normal_centers_all,
            title_cn="正常中心社交网络子图（全部中心）",
            out_path=out_normal_2hop,
        )

        draw_khop_subgraph(
            g_all=g_all,
            pred_labels=pred_all,
            centers=mal_centers_all,
            title_cn="恶意中心社交网络子图（全部中心）",
            out_path=out_mal_2hop,
        )

    # 4) 2-hop（恶意 Top-K 中心，按 mal_prob_all）
    out_mal_topk_2hop = os.path.join(NET_DIR, f"pred_malicious_centers_2hop_top{MAL_TOPK}.png")
    mal_centers_topk = select_centers_mal_topk(mal_prob_all, MAL_TOPK)

    draw_khop_subgraph(
        g_all=g_all,
        pred_labels=pred_all,
        centers=mal_centers_topk,
        title_cn=f"恶意中心社交网络子图（Top-{MAL_TOPK}）",
        out_path=out_mal_topk_2hop,
    )

    print(f"\n网络图已保存至 {NET_DIR}")
    print("- full_social_network_by_pred.png：全图")
    print("- pred_normal_social_graph.png：预测 normal 诱导子图（同类边）")
    print("- pred_malicious_social_graph.png：预测 malicious 诱导子图（同类边）")
    if DRAW_ALL_CENTERS_2HOP:
        print(f"- pred_normal_centers_2hop_all.png：normal 全中心 {HOPS}-hop 子图")
        print(f"- pred_malicious_centers_2hop_all.png：malicious 全中心 {HOPS}-hop 子图")
    print(f"- pred_malicious_centers_2hop_top{MAL_TOPK}.png：malicious Top-{MAL_TOPK} 中心 {HOPS}-hop 子图")

except Exception as e:
    print(f"网络图生成失败：{e}")

'''