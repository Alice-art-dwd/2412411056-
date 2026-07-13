import os
import pandas as pd
import numpy as np

# 定义桌面路径
base_path = r"C:\Users\HUAWEI\Desktop"
# 定义三个高度
heights = ["10m", "50m", "100m"]
# 定义原始文件的读取顺序
raw_file_order = ["train", "test", "val"]

for h in heights:
    print(f"\n==================== 开始处理 {h} 数据集 ====================")
    # 1. 纵向合并该高度下的所有原始数据
    df_list = []
    for status in raw_file_order:
        file_name = f"{status}_{h}.csv"
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            print(f" 警告：找不到文件 {file_path}，跳过该文件。")
            continue
        df_temp = pd.read_csv(file_path)

        # 【核心修复：直接精准删除多余的 Speed Avg 10m 列】
        if "Speed Avg 10m" in df_temp.columns:
            df_temp = df_temp.drop(columns=["Speed Avg 10m"])
            print(f" 成功从原始文件 {file_name} 中删除了多余的 'Speed Avg 10m' 列。")

        # 强制将 height 列修正为当前循环的实际高度数值
        h_numeric = int(h.replace("m", ""))
        df_temp["height"] = h_numeric
        df_list.append(df_temp)

    if not df_list:
        print(f" 未找到任何关于 {h} 的文件，跳过该高度。")
        continue

    df_merged = pd.concat(df_list, ignore_index=True)

    # 2. 转换为时间格式
    df_merged["Date & Time Stamp"] = pd.to_datetime(df_merged["Date & Time Stamp"])

    # 3. 【第一步：全局基础清洗】
    df_merged = df_merged.dropna(subset=["Date & Time Stamp"])
    df_merged = df_merged.drop_duplicates(subset=["Date & Time Stamp"])
    df_merged = df_merged.sort_values("Date & Time Stamp").reset_index(drop=True)

    # 保存合并后的完整大文件
    merged_output_path = os.path.join(base_path, f"merged_{h}.csv")
    df_merged.to_csv(merged_output_path, index=False)

    # 4. 【第二步：安全切分】按照 7:2:1 比例切分
    total_rows = len(df_merged)
    train_end = int(total_rows * 0.7)
    val_end = int(total_rows * 0.9)
    train_df = df_merged.iloc[:train_end].copy()
    val_df = df_merged.iloc[train_end:val_end].copy()
    test_df = df_merged.iloc[val_end:].copy()

    features = [col for col in df_merged.columns if col not in ["Date & Time Stamp", "height"]]

    # 5. 【第三步：独立处理各自的特征缺失值与异常值】
    # --- 5.1 处理训练集 (Train) ---
    for col in ["SpeedAvg", "SpeedMax"]:
        if col in train_df.columns: train_df.loc[train_df[col] < 0, col] = np.nan
    for col in ["HumidtyAvg", "HumityMax"]:
        if col in train_df.columns: train_df.loc[(train_df[col] < 0) | (train_df[col] > 100), col] = np.nan
    if "DirectionAvg" in train_df.columns:
        train_df.loc[(train_df["DirectionAvg"] < 0) | (train_df["DirectionAvg"] > 360), "DirectionAvg"] = np.nan

    stats_3sigma = {}
    for col in ["TemperatureAvg", "TemperatureMax", "PressureAvg", "PressureMax"]:
        if col in train_df.columns:
            mean = train_df[col].mean()
            std = train_df[col].std()
            stats_3sigma[col] = (mean, std)
            floor = mean - 3 * std
            ceil = mean + 3 * std
            train_df.loc[(train_df[col] < floor) | (train_df[col] > ceil), col] = np.nan

    train_df[features] = train_df[features].ffill().bfill()

    # --- 5.2 处理验证集 (Val) ---
    for col in ["SpeedAvg", "SpeedMax"]:
        if col in val_df.columns: val_df.loc[val_df[col] < 0, col] = np.nan
    for col in ["HumidtyAvg", "HumityMax"]:
        if col in val_df.columns: val_df.loc[(val_df[col] < 0) | (val_df[col] > 100), col] = np.nan
    if "DirectionAvg" in val_df.columns:
        val_df.loc[(val_df["DirectionAvg"] < 0) | (val_df["DirectionAvg"] > 360), "DirectionAvg"] = np.nan

    for col, (mean, std) in stats_3sigma.items():
        if col in val_df.columns:
            floor = mean - 3 * std
            ceil = mean + 3 * std
            val_df.loc[(val_df[col] < floor) | (val_df[col] > ceil), col] = np.nan
    val_df[features] = val_df[features].ffill().bfill()

    # --- 5.3 处理测试集 (Test) ---
    for col in ["SpeedAvg", "SpeedMax"]:
        if col in test_df.columns: test_df.loc[test_df[col] < 0, col] = np.nan
    for col in ["HumidtyAvg", "HumityMax"]:
        if col in test_df.columns: test_df.loc[(test_df[col] < 0) | (test_df[col] > 100), col] = np.nan
    if "DirectionAvg" in test_df.columns:
        test_df.loc[(test_df["DirectionAvg"] < 0) | (test_df["DirectionAvg"] > 360), "DirectionAvg"] = np.nan

    for col, (mean, std) in stats_3sigma.items():
        if col in test_df.columns:
            floor = mean - 3 * std
            ceil = mean + 3 * std
            test_df.loc[(test_df[col] < floor) | (test_df[col] > ceil), col] = np.nan
    test_df[features] = test_df[features].ffill().bfill()

    # 6. 保存处理好并重新划分的三个文件
    train_path = os.path.join(base_path, f"train_{h}_70pct.csv")
    val_path = os.path.join(base_path, f"val_{h}_20pct.csv")
    test_path = os.path.join(base_path, f"test_{h}_10pct.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f" {h} 数据集干净的 7:2:1 文件已重新覆盖生成！")

print("\n阶段1代码全部安全执行完毕！")