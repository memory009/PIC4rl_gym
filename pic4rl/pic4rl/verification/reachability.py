#!/usr/bin/env python3
"""
POLAR可达性计算核心模块
严格遵循论文算法实现
"""

import numpy as np
import sympy as sym
import time
from .taylor_model import (
    TaylorModel,
    TaylorArithmetic,
    BernsteinPolynomial,
    compute_tm_bounds,
    apply_activation
)


class POLARReachability:
    """
    POLAR可达性验证器
    
    论文核心算法实现：
    1. 输入区间 → Taylor模型
    2. 逐层传播（线性层 + 激活函数）
    3. 输出可达集
    """
    
    def __init__(self, weights, biases, max_action=0.5, bern_order=1, error_steps=4000):
        """
        Args:
            weights: list of np.ndarray - 权重矩阵 [W1, W2, W3, W4]
            biases: list of np.ndarray - 偏置向量 [b1, b2, b3, b4]
            max_action: float - 动作缩放系数（TD3的max_action）
            bern_order: int - Bernstein多项式阶数
            error_steps: int - 误差估计采样步数
        """
        self.weights = weights
        self.biases = biases
        self.max_action = max_action
        self.bern_order = bern_order
        self.error_steps = error_steps
        
        self.TA = TaylorArithmetic()
        self.BP = BernsteinPolynomial(error_steps=error_steps)
        
        # 网络结构
        self.num_layers = len(biases)
        self.input_dim = weights[0].shape[0]
        self.output_dim = biases[-1].shape[0]
        
        print(f"✅ POLAR验证器初始化:")
        print(f"  网络: {self.input_dim} → ... → {self.output_dim}")
        print(f"  层数: {self.num_layers}")
        print(f"  Max_action: {self.max_action}")
        print(f"  Bernstein阶数: {self.bern_order}")
    
    def compute_reachable_set(self, state, observation_error=0.01):
        """
        计算给定状态的动作可达集
        
        论文流程:
            Step 1: 构造输入Taylor模型 (归一化到[-1,1])
            Step 2: 逐层传播
            Step 3: 计算输出范围
        
        Args:
            state: np.ndarray (38,) - 机器人状态 [distance, angle, lidar_points...]
            observation_error: float - 观测误差范围
        
        Returns:
            action_ranges: list of [min, max] - 每个动作维度的可达集
            computation_time: float - 计算耗时
        """
        start_time = time.time()
        
        # Step 1: 构造输入Taylor模型
        print("\n[Step 1] 构造输入Taylor模型...")
        z_symbols = [sym.Symbol(f'z{i}') for i in range(self.input_dim)]
        TM_input = self._construct_input_tm(state, observation_error, z_symbols)
        
        # Step 2: 逐层传播
        print("\n[Step 2] 逐层传播...")
        TM_output = self._propagate_network(TM_input, z_symbols)
        
        # Step 3: 计算输出范围
        print("\n[Step 3] 计算输出范围...")
        action_ranges = []
        for i, tm in enumerate(TM_output):
            a, b = compute_tm_bounds(tm)
            # 应用max_action缩放
            a_scaled = a * self.max_action
            b_scaled = b * self.max_action
            action_ranges.append([a_scaled, b_scaled])
            print(f"  动作维度{i}: [{a_scaled:.6f}, {b_scaled:.6f}]")
        
        computation_time = time.time() - start_time
        print(f"\n✅ 可达性计算完成，耗时: {computation_time:.3f}s")
        
        return action_ranges, computation_time
    
    def _construct_input_tm(self, state, observation_error, z_symbols):
        """
        构造输入Taylor模型
        
        论文公式:
            xᵢ = state[i] + ε * zᵢ, zᵢ ∈ [-1, 1]
        
        这样归一化后，输入范围为:
            xᵢ ∈ [state[i] - ε, state[i] + ε]
        """
        TM_state = []
        for i in range(self.input_dim):
            # 关键：必须指定generators
            poly = sym.Poly(
                observation_error * z_symbols[i] + state[i],
                *z_symbols
            )
            TM_state.append(TaylorModel(poly, [0.0, 0.0]))
        
        print(f"  输入维度: {len(TM_state)}")
        print(f"  观测误差: ±{observation_error}")
        return TM_state
    
    def _propagate_network(self, TM_input, z_symbols):
        """
        逐层传播Taylor模型
        
        网络结构:
            [38] → [256, ReLU] → [128, ReLU] → [128, ReLU] → [2, Tanh]
        """
        TM_current = TM_input
        
        for layer_idx in range(self.num_layers):
            print(f"\n  Layer {layer_idx + 1}/{self.num_layers}:")
            
            # 获取权重和偏置
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            num_neurons = b.shape[0]
            
            print(f"    神经元数: {num_neurons}")
            
            TM_layer = []
            relu_stats = {'fully_active': 0, 'fully_inactive': 0, 'crossing_zero': 0}
            
            # 对每个神经元
            for neuron_idx in range(num_neurons):
                # 线性变换
                tm_linear = self.TA.weighted_sumforall(
                    TM_current,
                    W[:, neuron_idx],
                    b[neuron_idx]
                )
                
                # 激活函数
                is_hidden = (layer_idx < self.num_layers - 1)
                
                if is_hidden:
                    # 隐藏层：ReLU
                    tm_activated, stat = self._apply_relu(tm_linear, z_symbols)
                    relu_stats[stat] += 1
                else:
                    # 输出层：Tanh
                    tm_activated = self._apply_tanh(tm_linear)
                
                TM_layer.append(tm_activated)
            
            # 打印ReLU统计（仅隐藏层）
            if is_hidden:
                print(f"    ReLU统计:")
                print(f"      完全激活: {relu_stats['fully_active']}")
                print(f"      完全不激活: {relu_stats['fully_inactive']}")
                print(f"      跨越零点: {relu_stats['crossing_zero']}")
            
            TM_current = TM_layer
        
        return TM_current
    
    def _apply_relu(self, tm, z_symbols):
        """
        ReLU激活函数（三段式优化）
        
        论文 Equation (8):
            - b ≤ 0: ReLU(x) = 0
            - a ≥ 0: ReLU(x) = x
            - a < 0 < b: 使用Bernstein多项式
        """
        a, b = compute_tm_bounds(tm)
        
        if b <= 0:
            # 完全不激活
            zero_poly = sym.Poly(0, *z_symbols)
            return TaylorModel(zero_poly, [0, 0]), 'fully_inactive'
        
        elif a >= 0:
            # 完全激活
            return tm, 'fully_active'
        
        else:
            # 跨越零点，使用Bernstein多项式
            bern_poly = self.BP.approximate(a, b, self.bern_order, 'relu')
            bern_error = self.BP.compute_error(a, b, 'relu')
            tm_out = apply_activation(tm, bern_poly, bern_error, self.bern_order)
            return tm_out, 'crossing_zero'
    
    def _apply_tanh(self, tm):
        """
        Tanh激活函数
        
        输出层始终使用Bernstein多项式逼近
        """
        a, b = compute_tm_bounds(tm)
        
        bern_poly = self.BP.approximate(a, b, self.bern_order, 'tanh')
        bern_error = self.BP.compute_error(a, b, 'tanh')
        tm_out = apply_activation(tm, bern_poly, bern_error, self.bern_order)
        
        return tm_out


def analyze_safety(action_ranges, state):
    """
    分析动作可达集的安全性
    
    安全性判据（根据具体应用调整）：
    1. 可达集宽度不能太大（不确定性约束）
    2. 动作范围合理
    3. 如果接近障碍物，限制前进速度
    
    Args:
        action_ranges: [[v_min, v_max], [ω_min, ω_max]]
        state: 观测状态 [distance, angle, lidar...]
    
    Returns:
        is_safe: bool
        reasons: list of str
    """
    reasons = []
    is_safe = True
    
    # 提取激光雷达数据（假设前2维是distance和angle）
    if len(state) >= 10:
        lidar_readings = state[2:10]  # 前8个激光点
        min_laser = np.min(lidar_readings)
    else:
        min_laser = 10.0  # 默认安全
    
    # 检查1: 不确定性约束
    for i, (min_val, max_val) in enumerate(action_ranges):
        range_width = max_val - min_val
        if range_width > 0.3:  # 可达集宽度超过0.3
            is_safe = False
            reasons.append(f"动作{i}不确定性过大: {range_width:.4f} > 0.3")
    
    # 检查2: 碰撞风险
    if min_laser < 0.5:  # 距离障碍物小于0.5m
        linear_vel_range = action_ranges[0]
        if linear_vel_range[1] > 0.2:  # 可能快速前进
            is_safe = False
            reasons.append(f"接近障碍物({min_laser:.2f}m)但可能前进({linear_vel_range[1]:.3f})")
    
    # 检查3: 动作范围约束
    # 假设合理范围：线速度[-0.5, 0.5], 角速度[-1.0, 1.0]
    if action_ranges[0][0] < -0.6 or action_ranges[0][1] > 0.6:
        is_safe = False
        reasons.append(f"线速度超出安全范围: [{action_ranges[0][0]:.3f}, {action_ranges[0][1]:.3f}]")
    
    if action_ranges[1][0] < -1.2 or action_ranges[1][1] > 1.2:
        is_safe = False
        reasons.append(f"角速度超出安全范围: [{action_ranges[1][0]:.3f}, {action_ranges[1][1]:.3f}]")
    
    return is_safe, reasons