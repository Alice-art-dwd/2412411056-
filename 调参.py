import torch
import os

# 当模型在验证集上取得最好表现时，你运行下面这行命令
# torch.save(model.state_dict(), "best_lstm_model.pth")

learning_rates = [0.01, 0.001]
hidden_sizes = [32, 64]

print("====== 开始执行超参数网格搜索 ======")
for lr in learning_rates:
    for size in hidden_sizes:
        print(f"当前正在训练组合：学习率={lr}, 隐藏层维度={size}")

        # label_mse = 某个计算出来的误差值
        # print(f"该组合的测试集 MSE 为: {label_mse}")