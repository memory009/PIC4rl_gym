"""
激活函数定义
"""
import numpy as np
import torch


class ActivationFunctions:
    """激活函数集合（支持NumPy和PyTorch）"""
    
    @staticmethod
    def relu(x):
        """ReLU: max(0, x)"""
        if isinstance(x, torch.Tensor):
            return torch.relu(x)
        return np.maximum(0, x)
    
    @staticmethod
    def tanh(x):
        """Tanh: (e^x - e^-x) / (e^x + e^-x)"""
        if isinstance(x, torch.Tensor):
            return torch.tanh(x)
        return np.tanh(x)
    
    @staticmethod
    def linear(x):
        """Linear: f(x) = x"""
        return x
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid: 1 / (1 + e^-x)"""
        if isinstance(x, torch.Tensor):
            return torch.sigmoid(x)
        return 1.0 / (1.0 + np.exp(-x))


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("激活函数测试")
    print("="*60)
    
    test_values = [-2.0, -0.5, 0.0, 0.5, 2.0]
    
    print("\nNumPy版本:")
    for x in test_values:
        print(f"  x={x:5.1f} | ReLU={ActivationFunctions.relu(x):6.3f} | "
              f"Tanh={ActivationFunctions.tanh(x):6.3f}")
    
    print("\nPyTorch版本:")
    test_tensor = torch.tensor(test_values)
    relu_result = ActivationFunctions.relu(test_tensor)
    tanh_result = ActivationFunctions.tanh(test_tensor)
    
    for i, x in enumerate(test_values):
        print(f"  x={x:5.1f} | ReLU={relu_result[i]:6.3f} | "
              f"Tanh={tanh_result[i]:6.3f}")
    
    print("\n✅ 激活函数测试通过!")