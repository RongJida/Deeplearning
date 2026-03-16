import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重
w = 1.0
learning_rate = 0.01
epochs = 100

# 存储训练历史
w_history = []
loss_history = []
grad_history = []

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)
print('Predict (before traning)', 4, forward(4))

for epoch in range(epochs):
    for x, y in zip(x_data, y_data):
        grad_val = gradient(x, y)
        w = w - learning_rate * grad_val
        print("\tgrad", x, y, grad_val)
        l = loss(x, y)
        # 记录历史
        w_history.append(w)
        loss_history.append(l)
        grad_history.append(grad_val)
    print("progress:", epoch, "w=", w, "loss=", l)
print('Predict (after training)', 4, forward(4))

# 创建图形
plt.figure(figsize=(10, 4))

# 权重变化
plt.subplot(1, 2, 2)
plt.plot(w_history, 'r-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Weight (w)')
plt.title('权重变化')
plt.grid(True, alpha=0.3)
plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='理论最优 w=2.0')
plt.legend()


plt.tight_layout()
plt.show()
print(f"\n最终权重: {w:.4f}")
print(f"预测 x=4: {forward(4):.4f}")


