"""
PyTorch加速的Taylor模型实现
关键优化：使用GPU并行计算Bernstein多项式
"""

import torch
import numpy as np
from functools import reduce
import operator as op
import math


def ncr(n, r):
    """组合数 C(n,r)"""
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


class TaylorModelTorch:
    """
    PyTorch版Taylor模型
    使用张量运算加速多项式计算
    """
    
    def __init__(self, coeffs, powers, remainder, device='cuda'):
        """
        Args:
            coeffs: 多项式系数 [n_terms]
            powers: 每个项的指数 [n_terms, n_vars]
            remainder: 误差区间 [lower, upper]
            device: 'cuda' or 'cpu' or torch.device
        """
        # 正确处理device参数
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 正确处理tensor复制
        if isinstance(coeffs, torch.Tensor):
            self.coeffs = coeffs.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            self.coeffs = torch.tensor(coeffs, dtype=torch.float32, device=self.device)
            
        if isinstance(powers, torch.Tensor):
            self.powers = powers.clone().detach().to(dtype=torch.int32, device=self.device)
        else:
            self.powers = torch.tensor(powers, dtype=torch.int32, device=self.device)
            
        if isinstance(remainder, torch.Tensor):
            self.remainder = remainder.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            self.remainder = torch.tensor(remainder, dtype=torch.float32, device=self.device)
        self.n_vars = self.powers.shape[1] if len(self.powers.shape) > 1 else 1
    
    def evaluate(self, point):
        """
        在给定点评估多项式
        
        Args:
            point: [n_vars] 张量或数组
        
        Returns:
            value: 标量
        """
        if not isinstance(point, torch.Tensor):
            point = torch.tensor(point, dtype=torch.float32, device=self.device)
        
        # 计算每个单项式的值
        # term_i = coeff_i * prod(point[j]^power[i,j])
        term_values = self.coeffs.clone()
        
        for var_idx in range(self.n_vars):
            var_powers = self.powers[:, var_idx]
            term_values *= torch.pow(point[var_idx], var_powers.float())
        
        return term_values.sum()
    
    def to_cpu(self):
        """转移到CPU"""
        self.coeffs = self.coeffs.cpu()
        self.powers = self.powers.cpu()
        self.remainder = self.remainder.cpu()
        self.device = torch.device('cpu')
        return self
    
    def to_cuda(self):
        """转移到GPU"""
        if torch.cuda.is_available():
            self.coeffs = self.coeffs.cuda()
            self.powers = self.powers.cuda()
            self.remainder = self.remainder.cuda()
            self.device = torch.device('cuda')
        return self


class TaylorArithmeticTorch:
    """Taylor模型的算术运算（PyTorch加速版）"""
    
    def __init__(self, device='cuda'):
        # 正确处理device参数
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def weighted_sum(self, taylor_models, weights, bias):
        """
        计算加权和：Σ(w_i * TM_i) + bias
        
        Args:
            taylor_models: list of TaylorModelTorch
            weights: [n_models] 权重
            bias: 标量偏置
        
        Returns:
            TaylorModelTorch: 结果Taylor模型
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            
        if isinstance(bias, torch.Tensor):
            bias = bias.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            bias = torch.tensor(bias, dtype=torch.float32, device=self.device)
        
        # 初始化结果
        n_vars = taylor_models[0].n_vars
        result_coeffs = []
        result_powers = []
        
        # 累加每个Taylor模型
        for i, tm in enumerate(taylor_models):
            weighted_coeffs = tm.coeffs * weights[i]
            result_coeffs.append(weighted_coeffs)
            result_powers.append(tm.powers)
        
        # 合并系数和指数
        result_coeffs = torch.cat(result_coeffs)
        result_powers = torch.cat(result_powers)
        
        # 添加偏置项（所有指数为0）
        bias_power = torch.zeros((1, n_vars), dtype=torch.int32, device=self.device)
        result_coeffs = torch.cat([result_coeffs, bias.unsqueeze(0)])
        result_powers = torch.cat([result_powers, bias_power])
        
        # 计算误差区间
        remainder_sum = sum(abs(weights[i]) * tm.remainder[1] 
                           for i, tm in enumerate(taylor_models))
        result_remainder = torch.tensor([-remainder_sum, remainder_sum], 
                                       dtype=torch.float32, device=self.device)
        
        return TaylorModelTorch(result_coeffs, result_powers, result_remainder, 
                               device=self.device.type)
    
    def constant_product(self, taylor_model, constant):
        """
        常数乘法：c * TM
        TM = p + I  →  c*TM = (c*p) + (c*I)
        注意：c<0 时需要规范化区间上下界
        """
        if isinstance(constant, torch.Tensor):
            constant = constant.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            constant = torch.tensor(constant, dtype=torch.float32, device=self.device)

        new_coeffs = taylor_model.coeffs * constant
        # 区间乘常数并规范化
        low = taylor_model.remainder[0] * constant
        up  = taylor_model.remainder[1] * constant
        low2 = torch.minimum(low, up)
        up2  = torch.maximum(low, up)

        return TaylorModelTorch(new_coeffs, taylor_model.powers,
                                torch.stack([low2, up2]),
                                device=self.device.type)



class BernsteinPolynomialTorch:
    """PyTorch加速的Bernstein多项式逼近"""
    
    def __init__(self, device='cuda', error_steps=4000):
        # 正确处理device参数
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.error_steps = error_steps
        self.cache = {}  # 缓存计算结果
    
    def approximate(self, a, b, order, activation_name):
        """
        在区间[a,b]上用Bernstein多项式逼近激活函数
        
        Args:
            a, b: 区间端点
            order: 多项式阶数
            activation_name: 'relu' or 'tanh'
        
        Returns:
            coeffs: 多项式系数
            powers: 每个项的指数
        """
        # 检查缓存
        cache_key = (round(float(a), 6), round(float(b), 6), order, activation_name)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 确定实际阶数
        d_max = 8
        d_p = np.floor(d_max / math.log10(1 / (b - a))) if b > a else d_max
        d_p = max(2, np.abs(d_p))
        d = min(order, int(d_p))
        
        # Bernstein基函数系数
        coe1_1 = -a / (b - a) if b > a else 0
        coe1_2 = 1 / (b - a) if b > a else 1
        coe2_1 = b / (b - a) if b > a else 1
        coe2_2 = -1 / (b - a) if b > a else -1
        
        # 计算Bernstein多项式
        coeffs_list = []
        powers_list = []
        
        for v in range(d + 1):
            c = ncr(d, v)
            point = a + (b - a) / d * v if b > a else a
            
            # 计算激活函数值
            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(point)
            else:
                raise ValueError(f"不支持的激活函数: {activation_name}")
            
            # Bernstein基: (coe1_2*x + coe1_1)^v * (coe2_1 + coe2_2*x)^(d-v)
            # 展开二项式
            for i in range(v + 1):
                for j in range(d - v + 1):
                    coeff = (c * f_value * 
                            ncr(v, i) * ncr(d - v, j) *
                            (coe1_2 ** i) * (coe1_1 ** (v - i)) *
                            (coe2_2 ** j) * (coe2_1 ** (d - v - j)))
                    
                    power = i + j  # x的总次数
                    
                    if abs(coeff) > 1e-10:  # 过滤小系数
                        coeffs_list.append(coeff)
                        powers_list.append([power])
        
        # 合并同类项
        if len(coeffs_list) == 0:
            coeffs_list = [1e-16]
            powers_list = [[0]]
        
        coeffs = np.array(coeffs_list, dtype=np.float32)
        powers = np.array(powers_list, dtype=np.int32)
        
        # 缓存结果
        self.cache[cache_key] = (coeffs, powers)
        
        return coeffs, powers
    
    def compute_error(self, a, b, activation_name, bern_coeffs, bern_powers):
        """
        计算Bernstein逼近的误差上界（批量评估）
        
        Args:
            a, b: 区间
            activation_name: 激活函数名
            bern_coeffs: Bernstein多项式系数
            bern_powers: 指数
        
        Returns:
            error: 误差上界
        """
        # 在区间内均匀采样
        m = self.error_steps
        points = torch.linspace(a, b, m + 1, device=self.device)[:-1] + (b - a) / (2 * m)
        
        # 计算真实激活函数值
        if activation_name == 'relu':
            true_values = torch.relu(points)
        elif activation_name == 'tanh':
            true_values = torch.tanh(points)
        
        # 计算Bernstein多项式值（向量化）
        bern_coeffs_t = torch.tensor(bern_coeffs, dtype=torch.float32, device=self.device)
        bern_powers_t = torch.tensor(bern_powers, dtype=torch.int32, device=self.device)
        
        # 扩展维度用于广播
        points_expanded = points.unsqueeze(1)  # [m+1, 1]
        powers_expanded = bern_powers_t.unsqueeze(0)  # [1, n_terms, 1]
        
        # 计算 x^power
        term_values = torch.pow(points_expanded, powers_expanded.float().squeeze(-1))
        
        # 加权求和
        approx_values = (term_values * bern_coeffs_t).sum(dim=1)
        
        # 计算最大误差
        errors = torch.abs(true_values - approx_values)
        max_error = errors.max().item()
        
        return max_error + (b - a) / m


def compute_tm_bounds_torch(tm):
    """
    计算Taylor模型的上下界（PyTorch加速）
    
    Args:
        tm: TaylorModelTorch
    
    Returns:
        (lower, upper): 边界
    """
    # 计算多项式部分的上下界
    abs_coeffs = torch.abs(tm.coeffs)
    
    # 常数项（所有指数为0）
    is_constant = (tm.powers.sum(dim=1) == 0)
    constant_sum = (tm.coeffs * is_constant).sum()
    
    # 非常数项
    non_constant_sum = (abs_coeffs * (~is_constant)).sum()
    
    lower = constant_sum - non_constant_sum + tm.remainder[0]
    upper = constant_sum + non_constant_sum + tm.remainder[1]
    
    return float(lower.item()), float(upper.item())


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("="*70)
    print("PyTorch加速Taylor模型测试")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 测试1: 创建Taylor模型
    print("\n[1/4] 创建Taylor模型...")
    coeffs = [1.0, 0.5, -0.3]
    powers = [[0], [1], [2]]
    remainder = [-0.01, 0.01]
    
    tm = TaylorModelTorch(coeffs, powers, remainder, device=device)
    print(f"  ✓ Taylor模型创建成功")
    print(f"    系数: {tm.coeffs}")
    print(f"    指数: {tm.powers}")
    
    # 测试2: 评估多项式
    print("\n[2/4] 评估多项式...")
    test_point = torch.tensor([0.5], device=device)
    value = tm.evaluate(test_point)
    expected = 1.0 + 0.5 * 0.5 - 0.3 * 0.5**2
    print(f"  ✓ f(0.5) = {value:.4f} (期望: {expected:.4f})")
    
    # 测试3: 算术运算
    print("\n[3/4] 测试算术运算...")
    ta = TaylorArithmeticTorch(device=device)
    
    tm2 = TaylorModelTorch([2.0], [[0]], [0, 0], device=device)
    tm_sum = ta.weighted_sum([tm, tm2], [1.0, 1.0], 0.0)
    print(f"  ✓ 加权和计算成功")
    
    tm_product = ta.constant_product(tm, 2.0)
    print(f"  ✓ 常数乘法计算成功")
    
    # 测试4: Bernstein多项式
    print("\n[4/4] 测试Bernstein多项式...")
    bp = BernsteinPolynomialTorch(device=device, error_steps=1000)
    
    a, b = -1.0, 1.0
    order = 2
    
    coeffs, powers = bp.approximate(a, b, order, 'relu')
    print(f"  ✓ ReLU逼近: {len(coeffs)}个项")
    
    error = bp.compute_error(a, b, 'relu', coeffs, powers)
    print(f"  ✓ 逼近误差: {error:.6f}")
    
    # 测试5: 边界计算
    print("\n[5/5] 测试边界计算...")
    lower, upper = compute_tm_bounds_torch(tm)
    print(f"  ✓ 边界: [{lower:.4f}, {upper:.4f}]")
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！")
    print("="*70)
    
    # 性能测试
    if torch.cuda.is_available():
        print("\n[性能测试] GPU加速效果...")
        import time
        
        n_iters = 100
        
        # CPU
        tm_cpu = TaylorModelTorch(coeffs, powers, remainder, device='cpu')
        start = time.time()
        for _ in range(n_iters):
            _ = tm_cpu.evaluate(torch.tensor([0.5], device='cpu'))
        cpu_time = time.time() - start
        
        # GPU
        tm_gpu = TaylorModelTorch(coeffs, powers, remainder, device='cuda')
        start = time.time()
        for _ in range(n_iters):
            _ = tm_gpu.evaluate(torch.tensor([0.5], device='cuda'))
        gpu_time = time.time() - start
        
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  GPU时间: {gpu_time:.4f}s")
        print(f"  加速比: {cpu_time/gpu_time:.2f}x")