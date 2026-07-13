import os
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import joblib

# 我们在这里定义文件的基础路径
base_path = r"C:\Users\HUAWEI\Desktop"
data_dir = os.path.join(base_path, "风速预测数据中心")

heights = ["10m", "50m", "100m"]
tasks = ["single", "multiA", "multiB"]

# 我们在这里开始循环处理数据
for h in heights:
    for task in tasks:
        task_name = f"{task}_{h}"
        print(f"\n 开始训练 LightGBM 任务组合: {task_name}")

        # 我们在这里读取预先保存的矩阵文件
        try:
            X_train = np.load(os.path.join(data_dir, f"X_train_{task_name}.npy"))
            Y_train = np.load(os.path.join(data_dir, f"Y_train_{task_name}.npy"))
        except FileNotFoundError:
            print(f" 找不到 {task_name} 的数据文件，自动跳过。")
            continue

        # LightGBM 模型只能接受二维数据
        # 我们需要把特征数据的形状从三维压平成二维
        num_samples = X_train.shape[0]
        X_train_flat = X_train.reshape(num_samples, -1)

        # 我们在这里建立基础的 LightGBM 模型
        base_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )

        # 如果是多步预测，我们需要用多输出工具包装模型
        if task in ["multiA", "multiB"]:
            model = MultiOutputRegressor(base_model)
        else:
            # 如果是单步预测，我们需要保证标签是一维数组
            if len(Y_train.shape) > 1:
                Y_train = Y_train.squeeze()
            model = base_model

        # 我们开始训练模型
        print(f"   正在训练模型...")
        model.fit(X_train_flat, Y_train)

        # 我们把模型保存到指定的文件夹中
        model_save_path = os.path.join(data_dir, f"LightGBM_{task_name}.pth")
        joblib.dump(model, model_save_path)
        print(f"   LightGBM 模型已保存至: LightGBM_{task_name}.pth")

print("\n LightGBM 模型已经全部训练完毕！文件已经安全放入了“风速预测数据中心”文件夹中。")
