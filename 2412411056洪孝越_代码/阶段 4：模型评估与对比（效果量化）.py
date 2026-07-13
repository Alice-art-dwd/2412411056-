import os
import warnings

# 我们在这里直接隐藏所有的警告提示
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 我们在这里设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 我们在这里定义文件的基础路径
base_path = r"C:\Users\HUAWEI\Desktop"
data_dir = os.path.join(base_path, "风速预测数据中心")

# 我们选择运行代码的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 重新声明模型结构 ====================

class WindLinearRegression(nn.Module):
    def __init__(self, input_dim, lookback, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim * lookback, output_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class WindLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class WindTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, lookback):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model * 2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * lookback, output_dim)

    def forward(self, x):
        x = self.input_linear(x)
        out = self.transformer(x)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


# ============================================================

def inverse_scale_target(scaled_data, scaler, target_idx, num_features):
    if len(scaled_data.shape) == 1:
        scaled_data = scaled_data.reshape(-1, 1)
    N, steps = scaled_data.shape
    unscaled_data = np.zeros((N, steps))
    for t in range(steps):
        dummy = np.zeros((N, num_features))
        dummy[:, target_idx] = scaled_data[:, t]
        unscaled_data[:, t] = scaler.inverse_transform(dummy)[:, target_idx]
    return unscaled_data


heights = ["10m", "50m", "100m"]
tasks = ["single", "multiA", "multiB"]

all_results = []

for h in heights:
    train_path = os.path.join(base_path, f"train_{h}_70pct.csv")
    if not os.path.exists(train_path):
        print(f" 找不到 {h} 的训练集文件，无法建立缩放器。")
        continue
    df_train = pd.read_csv(train_path)
    features = [col for col in df_train.columns if col not in ["Date & Time Stamp", "height"]]
    target_idx = features.index("SpeedAvg")
    num_features = len(features)

    scaler = StandardScaler()
    scaler.fit(df_train[features])

    # 可视化 1：我们在这里绘制特征相关性热力图（每个高度画一张）
    print(f" 正在绘制 {h} 的特征相关性图...")
    plt.figure(figsize=(10, 8))
    corr = df_train[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"特征相关性热力图 ({h})")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"特征相关性_{h}.png"), dpi=300)
    plt.close()

    for task in tasks:
        task_name = f"{task}_{h}"

        try:
            X_test = np.load(os.path.join(data_dir, f"X_test_{task_name}.npy"))
            Y_test = np.load(os.path.join(data_dir, f"Y_test_{task_name}.npy"))
        except FileNotFoundError:
            continue

        if len(Y_test.shape) == 1:
            Y_test = Y_test.reshape(-1, 1)

        input_dim = X_test.shape[2]
        lookback = X_test.shape[1]
        output_dim = Y_test.shape[1]

        Y_test_unscaled = inverse_scale_target(Y_test, scaler, target_idx, num_features)

        # 可视化 2：我们在这里绘制测试集真实风速的分布图
        print(f"正在绘制 {task_name} 的数据集分布图...")
        plt.figure(figsize=(8, 5))
        sns.histplot(Y_test_unscaled.flatten(), kde=True, color='blue', bins=30)
        plt.title(f"测试集真实风速数据分布图 ({task_name})")
        plt.xlabel("真实风速 (m/s)")
        plt.ylabel("频数")
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"数据分布_{task_name}.png"), dpi=300)
        plt.close()

        model_info = {
            "Linear": os.path.join(data_dir, f"Linear_{task_name}.pth"),
            "LSTM": os.path.join(data_dir, f"LSTM_{task_name}.pth"),
            "Transformer": os.path.join(data_dir, f"Transformer_{task_name}.pth"),
            "LightGBM": os.path.join(data_dir, f"LightGBM_{task_name}.pth")
        }

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # 我们建立一个字典来存放每个模型的反归一化预测结果
        preds_dict = {}

        for model_name, model_path in model_info.items():
            if not os.path.exists(model_path):
                continue

            try:
                if model_name == "Linear":
                    model = WindLinearRegression(input_dim, lookback, output_dim).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        Y_pred = model(X_test_tensor).cpu().numpy()

                elif model_name == "LSTM":
                    model = WindLSTM(input_dim, hidden_dim=64, output_dim=output_dim).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        Y_pred = model(X_test_tensor).cpu().numpy()

                elif model_name == "Transformer":
                    model = WindTransformer(input_dim, d_model=32, nhead=4, num_layers=2, output_dim=output_dim,
                                            lookback=lookback).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        Y_pred = model(X_test_tensor).cpu().numpy()

                elif model_name == "LightGBM":
                    model = joblib.load(model_path)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    Y_pred = model.predict(X_test_flat)
                    if len(Y_pred.shape) == 1:
                        Y_pred = Y_pred.reshape(-1, 1)
            except Exception as e:
                print(f" 加载模型 {model_name}_{task_name} 失败: {e}")
                continue

            Y_pred_unscaled = inverse_scale_target(Y_pred, scaler, target_idx, num_features)
            preds_dict[model_name] = Y_pred_unscaled

            y_true_flat = Y_test_unscaled.flatten()
            y_pred_flat = Y_pred_unscaled.flatten()

            mse = mean_squared_error(y_true_flat, y_pred_flat)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            r2 = r2_score(y_true_flat, y_pred_flat)

            all_results.append({
                "高度": h,
                "任务": task,
                "模型": model_name,
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R²": round(r2, 4)
            })

        # 可视化 3：我们在这里绘制预测结果对比图（真实值 vs 多个模型的预测值）
        if len(preds_dict) > 0:
            print(f" 正在绘制 {task_name} 的模型预测对比图...")
            plt.figure(figsize=(12, 6))
            # 我们选取前 150 个时间点的数据进行展示，这样图表看起来很清晰
            sample_size = min(150, Y_test_unscaled.shape[0])

            # 绘制真实值
            plt.plot(Y_test_unscaled[:sample_size, 0], label="真实值 (True)", color='black', linewidth=2.5, zorder=5)

            # 绘制各个模型的预测值
            colors = {"Linear": "#1f77b4", "LSTM": "#ff7f0e", "Transformer": "#2ca02c", "LightGBM": "#d62728"}
            for model_name, y_pred_unscaled in preds_dict.items():
                plt.plot(y_pred_unscaled[:sample_size, 0], label=f"{model_name} 预测值", color=colors.get(model_name),
                         linestyle='--', linewidth=1.5)

            plt.title(f"不同模型预测结果对比图 ({task_name} - 前 {sample_size} 个样本点)")
            plt.xlabel("时间样本点")
            plt.ylabel("风速 (m/s)")
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, f"预测对比_{task_name}.png"), dpi=300)
            plt.close()

df_results = pd.DataFrame(all_results)
print("\n==================== 模型性能量化评估对比表 ====================")
print(df_results.to_string(index=False))

df_results.to_csv(os.path.join(data_dir, "模型性能对比结果表.csv"), index=False, encoding="utf-8-sig")
print(f"\n 所有的量化结果和对比图表都已经成功保存到了“风速预测数据中心”文件夹中。")