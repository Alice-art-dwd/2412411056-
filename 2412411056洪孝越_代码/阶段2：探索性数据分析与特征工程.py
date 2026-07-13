import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 定义桌面路径
base_path = r"C:\Users\HUAWEI\Desktop"
# 在桌面上创建一个独立的文件夹，用来存放所有全新生成的正确文件
output_dir = os.path.join(base_path, "风速预测数据中心")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

heights = ["10m", "50m", "100m"]

# 定义时间窗口参数（修改为老师要求的 8 小时历史窗口）
LOOKBACK = 48  # 8小时 = 48个点
FORECAST_SINGLE = 1  # 单步预测 = 1个点
FORECAST_A = 6  # 情况A：1小时 = 6个点
FORECAST_B = 96  # 情况B：16小时 = 96个点

# 设置画图的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_windows(data, lookback, forecast, target_idx):
    X_list = []
    Y_list = []
    total_len = len(data)
    for i in range(total_len - lookback - forecast + 1):
        x_window = data[i: i + lookback, :]
        if forecast == 1:
            y_window = data[i + lookback, target_idx]
        else:
            y_window = data[i + lookback: i + lookback + forecast, target_idx]
        X_list.append(x_window)
        Y_list.append(y_window)
    return np.array(X_list), np.array(Y_list)


# 开始循环处理每一个高度
for h in heights:
    print(f"\n==================== 开始执行 {h} 阶段2 ====================")
    train_path = os.path.join(base_path, f"train_{h}_70pct.csv")
    val_path = os.path.join(base_path, f"val_{h}_20pct.csv")
    test_path = os.path.join(base_path, f"test_{h}_10pct.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print(f" 错误：找不到 {h} 的清洗后文件，请先运行阶段1的代码。")
        continue

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    features = [col for col in df_train.columns if col not in ["Date & Time Stamp", "height"]]
    target_col = "SpeedAvg"
    target_idx = features.index(target_col)

    # 2. 画图分析数据
    print(" 正在生成全新的、真实独立的数据图表...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df_train[target_col], kde=True, color='blue')
    plt.title(f"{h} 训练集平均风速分布直方图")
    plt.xlabel("风速 (m/s)")
    plt.ylabel("频数")
    dist_img_path = os.path.join(output_dir, f"distribution_{h}.png")
    plt.savefig(dist_img_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr_matrix = df_train[features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{h} 特征相关性热力图")
    corr_img_path = os.path.join(output_dir, f"correlation_{h}.png")
    plt.savefig(corr_img_path, dpi=300)
    plt.close()
    print(f" 真实的 {h} 相关性热力图已安全保存，多余列已成功消失。")

    # 3. 特征转换与归一化
    print(" 正在进行特征归一化...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train[features])
    val_scaled = scaler.transform(df_val[features])
    test_scaled = scaler.transform(df_test[features])

    # 4. 构建时间窗口数据
    print(" 正在构建时间滑动窗口...")
    X_train_single, Y_train_single = create_windows(train_scaled, LOOKBACK, FORECAST_SINGLE, target_idx)
    X_val_single, Y_val_single = create_windows(val_scaled, LOOKBACK, FORECAST_SINGLE, target_idx)
    X_test_single, Y_test_single = create_windows(test_scaled, LOOKBACK, FORECAST_SINGLE, target_idx)

    X_train_multiA, Y_train_multiA = create_windows(train_scaled, LOOKBACK, FORECAST_A, target_idx)
    X_val_multiA, Y_val_multiA = create_windows(val_scaled, LOOKBACK, FORECAST_A, target_idx)
    X_test_multiA, Y_test_multiA = create_windows(test_scaled, LOOKBACK, FORECAST_A, target_idx)

    X_train_multiB, Y_train_multiB = create_windows(train_scaled, LOOKBACK, FORECAST_B, target_idx)
    X_val_multiB, Y_val_multiB = create_windows(val_scaled, LOOKBACK, FORECAST_B, target_idx)
    X_test_multiB, Y_test_multiB = create_windows(test_scaled, LOOKBACK, FORECAST_B, target_idx)

    # 5. 保存生成的矩阵文件
    print(" 正在保存正确的矩阵文件...")
    task_data = {
        f"single_{h}": (X_train_single, Y_train_single, X_val_single, Y_val_single, X_test_single, Y_test_single),
        f"multiA_{h}": (X_train_multiA, Y_train_multiA, X_val_multiA, Y_val_multiA, X_test_multiA, Y_test_multiA),
        f"multiB_{h}": (X_train_multiB, Y_train_multiB, X_val_multiB, Y_val_multiB, X_test_multiB, Y_test_multiB)
    }
    for task_name, arrays in task_data.items():
        np.save(os.path.join(output_dir, f"X_train_{task_name}.npy"), arrays[0])
        np.save(os.path.join(output_dir, f"Y_train_{task_name}.npy"), arrays[1])
        np.save(os.path.join(output_dir, f"X_val_{task_name}.npy"), arrays[2])
        np.save(os.path.join(output_dir, f"Y_val_{task_name}.npy"), arrays[3])
        np.save(os.path.join(output_dir, f"X_test_{task_name}.npy"), arrays[4])
        np.save(os.path.join(output_dir, f"Y_test_{task_name}.npy"), arrays[5])
    print(f" {h} 阶段2处理完成！")

print(f"\n 所有高维度的特征文件已经全部安全纠正并放入文件夹：{output_dir}")