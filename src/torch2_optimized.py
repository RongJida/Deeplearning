import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False

# 超参数配置
CONFIG = {
    'learning_rate': 0.01,
    'epochs': 100,
    'print_interval': 10,
    'theoretical_optimal': 2.0
}

# 训练数据
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])


def forward(x: np.ndarray, w: float) -> np.ndarray:
    """线性前向传播函数"""
    return x * w


def compute_cost(x: np.ndarray, y: np.ndarray, w: float) -> float:
    """计算损失函数（使用向量化计算）"""
    y_pred = forward(x, w)
    return np.mean((y_pred - y) ** 2)


def compute_gradient(x: np.ndarray, y: np.ndarray, w: float) -> float:
    """计算梯度（使用向量化计算）"""
    y_pred = forward(x, w)
    return np.mean(2 * x * (y_pred - y))


def train(x: np.ndarray, y: np.ndarray, config: dict) -> Tuple[float, List[float], List[float], List[float]]:
    """训练模型"""
    w = 1.0
    w_history = []
    cost_history = []
    grad_history = []
    
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    
    print(f'Predict (before training)前, 4: {forward(np.array([4]), w)[0]:.4f}')
    
    for epoch in range(epochs):
        cost_val = compute_cost(x, y, w)
        grad_val = compute_gradient(x, y, w)
        
        # 更新权重
        w -= learning_rate * grad_val
        
        # 记录历史
        w_history.append(w)
        cost_history.append(cost_val)
        grad_history.append(grad_val)
        
        # 打印训练信息
        if epoch % config['print_interval'] == 0:
            print(f'Epoch: {epoch:3d}, w={w:.6f}, loss={cost_val:.6f}, grad={grad_val:.6f}')
    
    print(f'Predict (after training)后, 4: {forward(np.array([4]), w)[0]:.4f}')
    
    return w, w_history, cost_history, grad_history


def plot_results(w_history: List[float], cost_history: List[float], config: dict):
    """可视化训练结果"""
    plt.figure(figsize=(12, 5))
    
    # 权重变化
    plt.subplot(1, 2, 1)
    plt.plot(w_history, 'r-', linewidth=2, label='训练权重')
    plt.xlabel('Epoch')
    plt.ylabel('Weight (w)')
    plt.title('权重变化')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=config['theoretical_optimal'], color='green', linestyle='--', alpha=0.5, label='理论最优')
    plt.legend()
    
    # 损失变化
    plt.subplot(1, 2, 2)
    plt.plot(cost_history, 'b-', linewidth=2, label='损失值')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('损失变化')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n最终权重: {w_history[-1]:.4f}")
    print(f"理论最优: {config['theoretical_optimal']}")
    print(f"预测 x=4: {forward(np.array([4]), w_history[-1])[0]:.4f}")


def main():
    """主函数"""
    # 训练模型
    w_final, w_history, cost_history, grad_history = train(x_data, y_data, CONFIG)
    
    # 可视化结果
    plot_results(w_history, cost_history, CONFIG)


if __name__ == '__main__':
    main()