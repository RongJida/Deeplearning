import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重
w = 1.0
learning_rate = 0.01
epochs = 100

# 存储训练历史
w_history = []
cost_history = []
grad_history = []


def forward(x):
    return x * w

def cost(xs, ys):
    total_cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        total_cost += (y_pred - y) ** 2
    return total_cost / len(xs)

def gradient(xs, ys):
    total_grad = 0
    for x, y in zip(xs, ys):
        total_grad += 2 * x * (x * w - y)
    return total_grad / len(xs)

print('Predict (before training)前', 4, forward(4))

for epoch in range(epochs):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= learning_rate * grad_val
     # 记录历史
    w_history.append(w)
    cost_history.append(cost_val)
    grad_history.append(grad_val)
    # 每10轮打印一次
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:3d}, w={w:.6f}, loss={cost_val:.6f}, grad={grad_val:.6f}')
print('Predict (after training)后 ', 4, forward(4))

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