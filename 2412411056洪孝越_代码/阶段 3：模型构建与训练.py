import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 你在这里定义文件的基础路径
base_path = r"C:\Users\HUAWEI\Desktop"
data_dir = os.path.join(base_path, "风速预测数据中心")

# 你在这里设置训练的参数
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.0005
PATIENCE = 5  # 如果连续5轮验证集损失没有下降，模型就会提前停止

# 你选择运行代码的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 当前代码正在使用的计算设备是: {device}")
print("💡 提示：如果这里显示的是 cpu，说明代码运行会非常慢。")


# ==================== 模型结构定义开始 ====================

class WindLinearRegression(nn.Module):
    def __init__(self, input_dim, lookback, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim * lookback, output_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out


class WindLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


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
        out = self.fc(out)
        return out


# ==================== 模型结构定义结束 ====================

def train_model(model, train_loader, val_loader, model_save_path, model_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        total_train_loss = train_loss / len(train_loader.dataset)
        total_val_loss = val_loss / len(val_loader.dataset)

        # 我在这里增加了每一轮的进度打印
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   [进度] {model_name} -> 轮次 {epoch+1}/{EPOCHS} | 训练集损失: {total_train_loss:.4f} | 验证集损失: {total_val_loss:.4f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 我在这里增加了早停机制
        if early_stop_counter >= PATIENCE:
            print(f"   [提示] {model_name} 的验证集损失连续 {PATIENCE} 轮没有下降，模型提前停止训练。")
            break


# 💡 调试建议：你可以先两行的取消下面注释。这样代码就只会跑一个组合。你可以用它来测试速度。
# heights = ["10m"]
# tasks = ["single"]

heights = ["10m", "50m", "100m"]
tasks = ["single", "multiA", "multiB"]

for h in heights:
    for task in tasks:
        task_name = f"{task}_{h}"
        print(f"\n🎬 开始训练任务组合: {task_name}")

        try:
            X_train = np.load(os.path.join(data_dir, f"X_train_{task_name}.npy"))
            Y_train = np.load(os.path.join(data_dir, f"Y_train_{task_name}.npy"))
            X_val = np.load(os.path.join(data_dir, f"X_val_{task_name}.npy"))
            Y_val = np.load(os.path.join(data_dir, f"Y_val_{task_name}.npy"))
        except FileNotFoundError:
            print(f" 找不到 {task_name} 的数据文件，自动跳过。")
            continue

        if len(Y_train.shape) == 1:
            Y_train = Y_train.reshape(-1, 1)
            Y_val = Y_val.reshape(-1, 1)

        input_dim = X_train.shape[2]
        lookback = X_train.shape[1]
        output_dim = Y_train.shape[1]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(Y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # ---- 1. 开始训练 Linear Regression ----
        linear_model = WindLinearRegression(input_dim, lookback, output_dim).to(device)
        linear_path = os.path.join(data_dir, f"Linear_{task_name}.pth")
        train_model(linear_model, train_loader, val_loader, linear_path, "Linear")
        print(f"   ✅ Linear 模型已保存")

        # ---- 2. 开始训练 LSTM ----
        lstm_model = WindLSTM(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, num_layers=2).to(device)
        lstm_path = os.path.join(data_dir, f"LSTM_{task_name}.pth")
        train_model(lstm_model, train_loader, val_loader, lstm_path, "LSTM")
        print(f"   ✅ LSTM 模型已保存")

        # ---- 3. 开始训练 Transformer ----
        transformer_model = WindTransformer(
            input_dim=input_dim, d_model=32, nhead=4, num_layers=2, output_dim=output_dim, lookback=lookback
        ).to(device)
        transformer_path = os.path.join(data_dir, f"Transformer_{task_name}.pth")
        train_model(transformer_model, train_loader, val_loader, transformer_path, "Transformer")
        print(f"   ✅ Transformer 模型已保存")

print("\n🎉 所有模型已经全部训练完毕！")