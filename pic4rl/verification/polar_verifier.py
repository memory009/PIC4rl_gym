"""
POLAR可达性验证器（核心模块）
使用PyTorch加速的Taylor模型
"""

import torch
import numpy as np
from .taylor_model_torch import (
    TaylorModelTorch,
    TaylorArithmeticTorch,
    BernsteinPolynomialTorch,
    compute_tm_bounds_torch
)
from .activation_functions import ActivationFunctions


class PolarVerifier:
    """
    POLAR可达性验证器
    
    功能：
    1. 从神经网络权重构建Taylor模型
    2. 逐层传播通过ReLU/Tanh激活函数
    3. 计算输出动作的可达集
    4. 判断安全性
    """
    
    def __init__(self, state_dim=38, action_dim=2, 
                 device='cuda', verbose=False):
        """
        Args:
            state_dim: 观测维度
            action_dim: 动作维度
            device: 'cuda' or 'cpu'
            verbose: 是否打印详细日志
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 确保 device 是 torch.device 对象
        if isinstance(device, str):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.device_str = str(self.device).replace('cuda:0', 'cuda')  # 用于传递给 TaylorModelTorch
        self.verbose = verbose
        
        # 初始化组件
        self.ta = TaylorArithmeticTorch(device=self.device)
        self.bp = BernsteinPolynomialTorch(device=self.device, error_steps=4000)
        
        # 统计信息
        self.stats = {
            'total_verifications': 0,
            'safe_count': 0,
            'avg_computation_time': 0
        }
        
        if self.verbose:
            print(f"✓ PolarVerifier初始化完成")
            print(f"  设备: {self.device}")
            print(f"  观测维度: {state_dim}")
            print(f"  动作维度: {action_dim}")
    
    def verify_safety(self, actor_weights, state, observation_error=0.01, 
                     bern_order=1, max_action=0.5):
        """
        POLAR安全性验证（主函数）
        
        Args:
            actor_weights: list of (W, b, activation)
                W: numpy array [input_dim, output_dim]
                b: numpy array [output_dim]
                activation: 'relu', 'tanh', or 'linear'
            state: numpy array [state_dim]
            observation_error: 观测误差范围
            bern_order: Bernstein多项式阶数
            max_action: 最大动作值（用于缩放输出）
        
        Returns:
            is_safe: bool
            ranges: [[v_min, v_max], [w_min, w_max]]
        """
        import time
        start_time = time.time()
        
        try:
            # 1. 构造输入Taylor模型
            TM_input = self._create_input_taylor_models(state, observation_error)
            
            # 2. 逐层传播
            for layer_idx, (W, b, activation) in enumerate(actor_weights):
                if self.verbose:
                    print(f"\n  [Layer {layer_idx+1}] {W.shape} | {activation}")
                
                TM_input = self._propagate_layer(
                    TM_input, W, b, activation, bern_order, max_action
                )
            
            # 3. 计算可达集
            ranges = self._compute_action_ranges(TM_input)
            
            # 4. 安全性判断
            is_safe = self._check_safety(ranges, state)
            
            # 更新统计
            self.stats['total_verifications'] += 1
            if is_safe:
                self.stats['safe_count'] += 1
            
            computation_time = time.time() - start_time
            self.stats['avg_computation_time'] = (
                (self.stats['avg_computation_time'] * (self.stats['total_verifications'] - 1) +
                 computation_time) / self.stats['total_verifications']
            )
            
            if self.verbose:
                print(f"\n  ✓ 验证完成: {'安全' if is_safe else '不安全'}")
                print(f"    可达集: v=[{ranges[0][0]:.3f}, {ranges[0][1]:.3f}], "
                      f"w=[{ranges[1][0]:.3f}, {ranges[1][1]:.3f}]")
                print(f"    耗时: {computation_time:.3f}s")
            
            return is_safe, ranges
            
        except Exception as e:
            print(f"✗ POLAR验证失败: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            
            # 返回保守的"不安全"
            return False, [[-1, 1], [-1, 1]]
    
    def _create_input_taylor_models(self, state, obs_error):
        """
        构造输入Taylor模型
        
        对于每个观测维度 s_i:
        TM_i = obs_error * z_i + s_i, remainder=[-ε, ε]
        
        Args:
            state: [state_dim]
            obs_error: 观测误差
        
        Returns:
            list of TaylorModelTorch
        """
        TM_list = []
        
        for i in range(self.state_dim):
            # 多项式: obs_error * z_i + s_i
            # 表示为: [obs_error, state[i]] * [z_i^1, z_i^0]
            coeffs = np.array([obs_error, state[i]], dtype=np.float32)
            powers = np.array([[1], [0]], dtype=np.int32)
            
            # 初始误差为0（主要误差在观测本身）
            remainder = np.array([0.0, 0.0], dtype=np.float32)
            
            tm = TaylorModelTorch(coeffs, powers, remainder, device=self.device_str)
            TM_list.append(tm)
        
        return TM_list
    
    def _propagate_layer(self, TM_input, W, b, activation, bern_order, max_action):
        """
        传播Taylor模型通过一层神经网络
        
        Args:
            TM_input: list of TaylorModelTorch [input_dim]
            W: numpy array [input_dim, output_dim]
            b: numpy array [output_dim]
            activation: 'relu', 'tanh', or 'linear'
            bern_order: Bernstein阶数
            max_action: 最大动作（仅输出层）
        
        Returns:
            list of TaylorModelTorch [output_dim]
        """
        output_dim = W.shape[1]
        TM_output = []
        
        for neuron_idx in range(output_dim):
            # 1. 线性变换: y = Σ(w_i * x_i) + b
            weights = W[:, neuron_idx]
            bias = b[neuron_idx]
            
            tm_neuron = self.ta.weighted_sum(TM_input, weights, bias)
            
            # 2. 应用激活函数
            if activation == 'linear':
                tm_activated = tm_neuron
            
            elif activation == 'relu':
                tm_activated = self._apply_relu(tm_neuron, bern_order)
            
            elif activation == 'tanh':
                tm_activated = self._apply_tanh(tm_neuron, bern_order)
                # 输出层需要缩放
                if max_action is not None:
                    tm_activated = self.ta.constant_product(tm_activated, max_action)
            
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
            
            TM_output.append(tm_activated)
        
        return TM_output
    
    def _apply_relu(self, tm, bern_order):
        """
        通过Bernstein多项式传播ReLU激活函数
        
        Args:
            tm: TaylorModelTorch
            bern_order: 多项式阶数
        
        Returns:
            TaylorModelTorch
        """
        # 计算输入区间
        a, b = compute_tm_bounds_torch(tm)
        
        # 情况1: 完全在正区域 [a, b] ⊂ [0, ∞)
        if a >= 0:
            return tm
        
        # 情况2: 完全在负区域 [a, b] ⊂ (-∞, 0]
        if b <= 0:
            # 返回零多项式
            zero_coeffs = np.array([0.0], dtype=np.float32)
            zero_powers = np.array([[0]], dtype=np.int32)
            zero_remainder = np.array([0.0, 0.0], dtype=np.float32)
            return TaylorModelTorch(zero_coeffs, zero_powers, zero_remainder, 
                                   device=self.device_str)
        
        # 情况3: 跨越零点，需要Bernstein逼近
        bern_coeffs, bern_powers = self.bp.approximate(a, b, bern_order, 'relu')
        bern_error = self.bp.compute_error(a, b, 'relu', bern_coeffs, bern_powers)
        
        return self._apply_bernstein_activation(tm, bern_coeffs, bern_powers, bern_error)
    
    def _apply_tanh(self, tm, bern_order):
        """通过Bernstein多项式传播Tanh激活函数"""
        a, b = compute_tm_bounds_torch(tm)
        
        bern_coeffs, bern_powers = self.bp.approximate(a, b, bern_order, 'tanh')
        bern_error = self.bp.compute_error(a, b, 'tanh', bern_coeffs, bern_powers)
        
        return self._apply_bernstein_activation(tm, bern_coeffs, bern_powers, bern_error)
    
    # def _apply_bernstein_activation(self, tm, bern_coeffs, bern_powers, bern_error):
    #     """
    #     应用Bernstein多项式激活函数
        
    #     核心思想：
    #     1. 用Bernstein多项式 B(x) 逼近激活函数 f(x)
    #     2. 计算 B(TM_input)
    #     3. 添加逼近误差和截断误差
    #     """
    #     # 将输入TM的系数和指数提取到CPU（符号计算）
    #     input_coeffs = tm.coeffs.cpu().numpy()
    #     input_powers = tm.powers.cpu().numpy()
        
    #     # 1. 合成多项式: B(p(z))
    #     # 这里简化实现：仅保留一阶项
    #     # 完整实现需要多项式合成算法
        
    #     # 简化方案：使用线性逼近
    #     # B(p(z)) ≈ B(p_0) + B'(p_0) * (p(z) - p_0)
        
    #     # 计算p_0（常数项）
    #     is_constant = (input_powers.sum(axis=1) == 0)
    #     p_0 = input_coeffs[is_constant].sum() if is_constant.any() else 0.0
        
    #     # 计算B(p_0)
    #     bern_at_p0 = 0.0
    #     for coeff, power in zip(bern_coeffs, bern_powers):
    #         bern_at_p0 += coeff * (p_0 ** power[0])
        
    #     # 新的Taylor模型
    #     # 简化：直接使用Bernstein多项式的系数
    #     result_coeffs = bern_coeffs
    #     result_powers = bern_powers
        
    #     # 计算总误差
    #     # 1. Bernstein逼近误差
    #     # 2. 输入Taylor模型的传播误差
    #     input_error = tm.remainder[1].cpu().item()
        
    #     # 简化误差估计
    #     total_error = bern_error + 2 * input_error
    #     result_remainder = np.array([-total_error, total_error], dtype=np.float32)
        
    #     return TaylorModelTorch(result_coeffs, result_powers, result_remainder,
    #                            device=self.device)

    def _apply_bernstein_activation(self, tm, bern_coeffs, bern_powers, bern_error):
        """
        按论文式(5)： TM_out = p_sigma(TM_in) + Int(r_k) + I_sigma
        这里用 k=1 的线性化展开：p(TM) ≈ p(p0) + p'(p0)*(TM - p0)
        截断余项 Int(r_k) 用二阶导数的上界与“半径”估计；I_sigma=[-ε,+ε] 来自Bernstein误差。
        """
        # ===== 1) 取输入 TM 的中心 p0 与“半径”rad =====
        # p0 = 常数项之和（多项式常数）+ 余项中心，这里按论文将余项放入 remainder，不加到中心
        input_coeffs = tm.coeffs
        input_powers = tm.powers
        is_const = (input_powers.sum(dim=1) == 0)
        p0 = input_coeffs[is_const].sum() if is_const.any() else torch.tensor(0.0, device=self.device)

        # (a,b) 为 TM 的保守上下界（仅用于界定二导数范围与 I_sigma 的区间） 
        a, b = compute_tm_bounds_torch(tm)  # 已有函数
        a = float(a); b = float(b)

        # ===== 2) 计算 p(p0)、p'(p0)、max|p''(x)| on [a,b] =====
        # 你的 Bernstein 返回的是“在 x 上的幂基表示”： sum c_i * x^{pow_i}
        # 我们对这个一元多项式直接求导（解析），在区间 [a,b] 采样上界。
        # p(x)
        coeff = torch.tensor(bern_coeffs, dtype=torch.float32, device=self.device)
        power = torch.tensor(bern_powers, dtype=torch.int32,   device=self.device).squeeze(-1)  # [n]
        # p(p0)
        p0_pow = torch.pow(p0, power.float())
        p_at_p0 = (coeff * p0_pow).sum()

        # p'(x) 系数/幂
        d1_coeff = coeff * power.float()
        d1_power = torch.clamp(power.int() - 1, min=0)
        # p'(p0)
        d1_val = (d1_coeff * torch.pow(p0, d1_power.float())).sum()

        # p''(x)
        d2_coeff = d1_coeff * d1_power.float()
        d2_power = torch.clamp(d1_power - 1, min=0)

        # 在 [a,b] 上采样 max|p''(x)|（少量采样即可，误差进入 remainder）
        xs = torch.linspace(a, b, steps=64, device=self.device)
        if xs.numel() == 0:
            xs = torch.tensor([a], device=self.device)
        d2_vals = (d2_coeff.view(1, -1) * torch.pow(xs.view(-1, 1), d2_power.view(1, -1).float())).sum(dim=1)
        max_abs_d2 = torch.max(torch.abs(d2_vals)).item()

        # ===== 3) 构造 ΔTM = TM - p0（把常数项“抽走”，保留一次及以上项）=====
        # 系数：常数项减去 p0 后变 0；其它项不变
        coeffs = tm.coeffs.clone()
        if is_const.any():
            # 将常数项的系数合并成一个常数 monomial 再减 p0（或直接把所有常数系数清零，再单独加一个常数 TM）
            const_sum = input_coeffs[is_const].sum()
            # 把所有常数项清零
            coeffs = coeffs.clone()
            coeffs[is_const] = 0.0
        # ΔTM 的 remainder 与 TM 相同（区间平移不改半径）
        delta_tm = TaylorModelTorch(coeffs, tm.powers, tm.remainder, device=self.device_str)

        # ===== 4) 线性化合成： p(TM) ≈ p(p0) + p'(p0) * ΔTM =====
        # 常数 TM
        n_vars = delta_tm.n_vars
        const_power = torch.zeros((1, n_vars), dtype=torch.int32, device=self.device)
        const_tm = TaylorModelTorch(torch.tensor([p_at_p0], device=self.device),
                                    const_power,
                                    torch.tensor([0.0, 0.0], device=self.device),
                                    device=self.device_str)
        # 斜率缩放
        linear_tm = self.ta.constant_product(delta_tm, d1_val)

        # 线性化多项式部分：const + slope*ΔTM
        p_of_tm = self.ta.weighted_sum([const_tm, linear_tm], [1.0, 1.0], bias=0.0)

        # ===== 5) 余项：Int(r_k)（截断） + I_sigma（Bernstein） + 斜率 × 输入余项传播 =====
        # 5.1 Bernstein 逼近误差 I_sigma
        I_sigma_low = -float(bern_error)
        I_sigma_up  = +float(bern_error)

        # 5.2 线性化截断（二阶项起）： 0.5 * max|p''(x)| * rad^2
        # 这里 rad 估计为 ΔTM 的上界幅度（不含 remainder 中心，只取半径）
        # 用现有的 compute_tm_bounds_torch 给 ΔTM 一个区间，再取半径
        da, db = compute_tm_bounds_torch(delta_tm)
        rad = 0.5 * (db - da)
        trunc_err = 0.5 * max_abs_d2 * (rad ** 2)

        # 5.3 输入 remainder 通过斜率传播：|p'(p0)| * I_in
        I_in_low  = float(tm.remainder[0].item())
        I_in_up   = float(tm.remainder[1].item())
        slope = float(abs(d1_val.item()))
        prop_low = slope * I_in_low
        prop_up  = slope * I_in_up
        # 由于 I_in_low 可能为负，规范化
        prop_L = min(prop_low, prop_up)
        prop_U = max(prop_low, prop_up)

        # 合并余项并规范化
        rem_L = I_sigma_low - abs(trunc_err) + prop_L
        rem_U = I_sigma_up  + abs(trunc_err) + prop_U
        remainder = torch.tensor([rem_L, rem_U], dtype=torch.float32, device=self.device)

        # 输出 TM：线性化多项式 + 合并 remainder
        return TaylorModelTorch(p_of_tm.coeffs, p_of_tm.powers, remainder, device=self.device_str)
    
    def _compute_action_ranges(self, TM_output):
        """
        计算输出动作的可达集
        
        Args:
            TM_output: list of TaylorModelTorch [action_dim]
        
        Returns:
            ranges: [[v_min, v_max], [w_min, w_max]]
        """
        ranges = []
        
        for tm in TM_output:
            lower, upper = compute_tm_bounds_torch(tm)
            ranges.append([lower, upper])
        
        return ranges
    
    def _check_safety(self, ranges, state):
        """
        判断可达集是否安全
        
        Args:
            ranges: [[v_min, v_max], [w_min, w_max]]
            state: 当前观测 [state_dim]
        
        Returns:
            is_safe: bool
        """
        # 规则1: 可达集宽度不能太大（不确定性检查）
        v_width = ranges[0][1] - ranges[0][0]
        w_width = ranges[1][1] - ranges[1][0]
        
        if v_width > 1.5 or w_width > 1.5:
            if self.verbose:
                print(f"    ✗ 不安全: 可达集过宽 (v={v_width:.3f}, w={w_width:.3f})")
            return False
        
        # 规则2: 检查动作范围是否合理
        # 线速度应在 [-0.5, 0.5] 范围内
        if ranges[0][0] < -0.6 or ranges[0][1] > 0.6:
            if self.verbose:
                print(f"    ✗ 不安全: 线速度超限 [{ranges[0][0]:.3f}, {ranges[0][1]:.3f}]")
            return False
        
        # 角速度应在 [-1.0, 1.0] 范围内
        if ranges[1][0] < -1.1 or ranges[1][1] > 1.1:
            if self.verbose:
                print(f"    ✗ 不安全: 角速度超限 [{ranges[1][0]:.3f}, {ranges[1][1]:.3f}]")
            return False
        
        # 规则3: 碰撞风险检查（基于激光雷达）
        # state[2:38] 是36个激光点（归一化到[0, 1]）
        if len(state) >= 38:
            laser_readings = state[2:38] * 10.0  # 反归一化到[0, 10]米
            min_laser = np.min(laser_readings)
            
            # 如果距离障碍物很近且可能前进
            if min_laser < 0.5 and ranges[0][1] > 0.3:
                if self.verbose:
                    print(f"    ✗ 不安全: 碰撞风险 (最近障碍={min_laser:.2f}m, "
                          f"最大前进速度={ranges[0][1]:.3f})")
                return False
        
        return True
    
    def get_statistics(self):
        """获取验证统计信息"""
        if self.stats['total_verifications'] > 0:
            safety_rate = self.stats['safe_count'] / self.stats['total_verifications']
        else:
            safety_rate = 0.0
        
        return {
            'total_verifications': self.stats['total_verifications'],
            'safe_count': self.stats['safe_count'],
            'safety_rate': safety_rate,
            'avg_computation_time': self.stats['avg_computation_time']
        }


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("="*70)
    print("POLAR验证器测试")
    print("="*70)
    
    # 创建验证器
    print("\n[1/3] 创建验证器...")
    verifier = PolarVerifier(state_dim=38, action_dim=2, verbose=True)
    print("  ✓ 验证器创建成功")
    
    # 创建模拟网络权重
    print("\n[2/3] 创建模拟Actor网络...")
    actor_weights = [
        # Layer 1: 38 → 256, ReLU
        (np.random.randn(38, 256).astype(np.float32) * 0.1,
         np.random.randn(256).astype(np.float32) * 0.1,
         'relu'),
        
        # Layer 2: 256 → 128, ReLU
        (np.random.randn(256, 128).astype(np.float32) * 0.1,
         np.random.randn(128).astype(np.float32) * 0.1,
         'relu'),
        
        # Layer 3: 128 → 128, ReLU
        (np.random.randn(128, 128).astype(np.float32) * 0.1,
         np.random.randn(128).astype(np.float32) * 0.1,
         'relu'),
        
        # Output: 128 → 2, Tanh
        (np.random.randn(128, 2).astype(np.float32) * 0.1,
         np.random.randn(2).astype(np.float32) * 0.1,
         'tanh')
    ]
    print("  ✓ 网络结构: 38→256→128→128→2")
    
    # 创建模拟观测
    print("\n[3/3] 执行POLAR验证...")
    state = np.random.rand(38).astype(np.float32)
    state[0] = 0.5  # 距离目标 (归一化)
    state[1] = 0.2  # 角度 (归一化)
    state[2:38] = 0.7  # 激光雷达读数
    
    is_safe, ranges = verifier.verify_safety(
        actor_weights,
        state,
        observation_error=0.01,
        bern_order=1,
        max_action=0.5
    )
    
    # 显示结果
    print("\n" + "="*70)
    print("验证结果")
    print("="*70)
    print(f"  安全性: {'✅ 安全' if is_safe else '⚠️ 不安全'}")
    print(f"  线速度可达集: [{ranges[0][0]:.4f}, {ranges[0][1]:.4f}]")
    print(f"  角速度可达集: [{ranges[1][0]:.4f}, {ranges[1][1]:.4f}]")
    print(f"  可达集宽度: v={ranges[0][1]-ranges[0][0]:.4f}, "
          f"w={ranges[1][1]-ranges[1][0]:.4f}")
    
    # 统计信息
    stats = verifier.get_statistics()
    print(f"\n统计信息:")
    print(f"  总验证次数: {stats['total_verifications']}")
    print(f"  安全率: {stats['safety_rate']:.2%}")
    print(f"  平均计算时间: {stats['avg_computation_time']:.3f}秒")
    
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)