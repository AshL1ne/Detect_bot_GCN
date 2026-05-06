import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import pandas as pd

# -------------------- 配置 --------------------
GRAPH_PATH = "../output/graph_data.pt"          # 与 gcn_train.py 保持一致
MODEL_PATH = "../output/GCN_model.pt"
USER_CSV   = "../output/users_raw.csv"          # 用于获取 user_id 顺序
THRESHOLD  = 0.5                                # 可修改为 0.7 等
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- 模型定义（必须与训练时相同） --------------------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# -------------------- FastAPI 实例（全局定义，不涉及加载逻辑） --------------------
app = FastAPI(title="GCN Malicious User Detector", version="1.0")

# 全局预测结果字典（启动后填充）
user_predictions = {}

class BatchRequest(BaseModel):
    user_ids: list[str]

@app.get("/predict/{user_id}")
async def predict_single(user_id: str):
    if user_id not in user_predictions:
        raise HTTPException(status_code=404, detail="用户 ID 不在图中")
    return {
        "user_id": user_id,
        **user_predictions[user_id]
    }

@app.post("/predict_batch")
async def predict_batch(req: BatchRequest):
    results = []
    for uid in req.user_ids:
        if uid in user_predictions:
            results.append({
                "user_id": uid,
                **user_predictions[uid]
            })
        else:
            results.append({
                "user_id": uid,
                "error": "not found"
            })
    return {"results": results}

@app.get("/health")
async def health():
    return {"status": "ok", "cached_users": len(user_predictions)}

# -------------------- 核心：资源加载+服务启动+异常捕获 --------------------
if __name__ == "__main__":
    import uvicorn
    try:
        # 1. 先检查文件是否存在，提前发现路径错误
        print("===== 检查资源文件 =====")
        for file_path, name in zip([GRAPH_PATH, MODEL_PATH, USER_CSV], ["图数据", "模型", "用户CSV"]):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{name}文件不存在！路径：{os.path.abspath(file_path)}")
            print(f"{name}文件存在：{os.path.abspath(file_path)}")

        # 2. 加载图数据
        print("\n===== 加载图数据 =====")
        data = torch.load(GRAPH_PATH, map_location=DEVICE, weights_only=False)
        print(f"图数据加载完成，节点数：{data.num_nodes}，特征维度：{data.num_features}")

        # 3. 加载模型
        print("\n===== 加载模型 =====")
        model = GCN(
            in_dim=data.num_features,
            hidden_dim=64,
            out_dim=data.num_classes,
            dropout=0.5
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("模型加载完成，已切换为推理模式")

        # 4. 全图推理
        print("\n===== 执行全图推理 =====")
        data = data.to(DEVICE)
        with torch.no_grad():
            logits = model(data)
            prob = logits.exp()[:, 1]        # 恶意类概率
            mal_prob = prob.cpu().numpy()
        print(f"全图推理完成，共生成 {len(mal_prob)} 个节点的恶意概率")

        # 5. 构建用户ID-预测结果映射
        print("\n===== 构建用户预测结果 =====")
        users_raw = pd.read_csv(USER_CSV, dtype={'_id': str})
        user_id_list = users_raw['_id'].tolist()

        if len(user_id_list) != len(mal_prob):
            raise ValueError(f"用户数量与节点数量不匹配！用户数：{len(user_id_list)}，节点数：{len(mal_prob)}")

        for idx, uid in enumerate(user_id_list):
            user_predictions[uid] = {
                "mal_prob": float(mal_prob[idx]),
                "is_malicious": int(mal_prob[idx] >= THRESHOLD)
            }
        print(f"已缓存 {len(user_predictions)} 个用户的预测结果")

        # 6. 启动服务
        print("\n===== 启动API服务 =====")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    # 捕获所有异常，打印详细错误信息
    except Exception as e:
        print(f"\n服务启动失败！错误原因：{e}")
        print("===== 详细错误堆栈 =====")
        import traceback
        traceback.print_exc()

#启动：uvicorn gcn_api:app --host 0.0.0.0 --port 8000