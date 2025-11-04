# -*- coding: utf-8 -*-
"""
Taylor模型单元测试
验证PyTorch实现的正确性
"""

import sys
import os
# 获取绝对路径，确保正确添加到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
from pic4rl.verification.taylor_model_torch import (
    TaylorModelTorch,
    TaylorArithmeticTorch,
    BernsteinPolynomialTorch,
    compute_tm_bounds_torch
)


def test_taylor_model_creation():
    """测试1: Taylor模型创建"""
    print("\n[测试1] Taylor模型创建...")
    
    # 创建简单多项式: f(x) = 2 + 3x + x^2
    coeffs = [2.0, 3.0, 1.0]
    powers = [[0], [1], [2]]
    remainder = [-0.01, 0.01]
    
    tm = TaylorModelTorch(coeffs, powers, remainder)
    
    # 在x=2处评估
    value = tm.evaluate(torch.tensor([2.0]))
    expected = 2 + 3*2 + 2**2  # = 12
    
    assert abs(value.item() - expected) < 0.01, f"评估错误: {value} vs {expected}"
    
    print(f"  ✓ 创建成功: f(2) = {value.item():.3f} (期望: {expected})")


def test_taylor_arithmetic():
    """测试2: Taylor算术运算"""
    print("\n[测试2] Taylor算术运算...")
    
    ta = TaylorArithmeticTorch()
    
    # 创建两个TM
    tm1 = TaylorModelTorch([1.0, 2.0], [[0], [1]], [0, 0])
    tm2 = TaylorModelTorch([3.0, -1.0], [[0], [1]], [0, 0])
    
    # 加权和: 2*tm1 + 3*tm2 + 5
    result = ta.weighted_sum([tm1, tm2], [2.0, 3.0], 5.0)
    
    # 在x=1处评估
    value = result.evaluate(torch.tensor([1.0]))
    expected = 2*(1+2*1) + 3*(3-1*1) + 5  # = 2*3 + 3*2 + 5 = 17
    
    assert abs(value.item() - expected) < 0.01, f"加权和错误: {value} vs {expected}"
    
    print(f"  ✓ 加权和: f(1) = {value.item():.3f} (期望: {expected})")
    
    # 常数乘法
    result2 = ta.constant_product(tm1, 5.0)
    value2 = result2.evaluate(torch.tensor([1.0]))
    expected2 = 5 * (1 + 2*1)  # = 15
    
    assert abs(value2.item() - expected2) < 0.01, f"常数乘法错误: {value2} vs {expected2}"
    
    print(f"  ✓ 常数乘法: 5*f(1) = {value2.item():.3f} (期望: {expected2})")


def test_bernstein_approximation():
    """测试3: Bernstein多项式逼近"""
    print("\n[测试3] Bernstein多项式逼近...")
    
    bp = BernsteinPolynomialTorch(error_steps=1000)
    
    # 测试ReLU逼近
    a, b = -1.0, 2.0
    order = 3
    
    coeffs, powers = bp.approximate(a, b, order, 'relu')
    error = bp.compute_error(a, b, 'relu', coeffs, powers)
    
    print(f"  ✓ ReLU逼近 [{a}, {b}]:")
    print(f"    项数: {len(coeffs)}")
    print(f"    误差: {error:.6f}")
    
    # 验证逼近质量
    test_points = torch.linspace(a, b, 100)
    true_values = torch.relu(test_points)
    
    # 评估Bernstein多项式
    coeffs_t = torch.tensor(coeffs)
    powers_t = torch.tensor(powers)
    
    approx_values = []
    for x in test_points:
        val = sum(c * (x ** p[0]) for c, p in zip(coeffs_t, powers_t))
        approx_values.append(val)
    
    approx_values = torch.stack(approx_values)
    max_diff = (true_values - approx_values).abs().max()
    
    print(f"    实际最大误差: {max_diff.item():.6f}")
    
    assert max_diff < error + 0.01, f"误差估计不准确: {max_diff} > {error}"


def test_bounds_computation():
    """测试4: 边界计算"""
    print("\n[测试4] 边界计算...")
    
    # 创建TM: f(x) = 1 + 2x, x ∈ [-1, 1]
    coeffs = [1.0, 2.0]
    powers = [[0], [1]]
    remainder = [0.0, 0.0]
    
    tm = TaylorModelTorch(coeffs, powers, remainder)
    
    lower, upper = compute_tm_bounds_torch(tm)
    
    # 理论边界: f(-1) = -1, f(1) = 3
    expected_lower = -1.0
    expected_upper = 3.0
    
    print(f"  ✓ 边界: [{lower:.3f}, {upper:.3f}]")
    print(f"    期望: [{expected_lower:.3f}, {expected_upper:.3f}]")
    
    assert abs(lower - expected_lower) < 0.01, f"下界错误: {lower} vs {expected_lower}"
    assert abs(upper - expected_upper) < 0.01, f"上界错误: {upper} vs {expected_upper}"


def test_gpu_acceleration():
    """测试5: GPU加速效果"""
    print("\n[测试5] GPU加速...")
    
    if not torch.cuda.is_available():
        print("  ⚠️  GPU不可用，跳过此测试")
        return
    
    import time
    
    # 创建较大的Taylor模型
    n_terms = 100
    coeffs = np.random.randn(n_terms).astype(np.float32)
    powers = np.random.randint(0, 3, (n_terms, 1)).astype(np.int32)
    remainder = np.array([0.0, 0.0], dtype=np.float32)
    
    # CPU版本
    tm_cpu = TaylorModelTorch(coeffs, powers, remainder, device='cpu')
    test_point_cpu = torch.randn(1, device='cpu')
    
    start = time.time()
    for _ in range(1000):
        _ = tm_cpu.evaluate(test_point_cpu)
    cpu_time = time.time() - start
    
    # GPU版本
    tm_gpu = TaylorModelTorch(coeffs, powers, remainder, device='cuda')
    test_point_gpu = torch.randn(1, device='cuda')
    
    # 预热
    for _ in range(10):
        _ = tm_gpu.evaluate(test_point_gpu)
    
    start = time.time()
    for _ in range(1000):
        _ = tm_gpu.evaluate(test_point_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    
    print(f"  ✓ CPU时间: {cpu_time:.4f}s")
    print(f"  ✓ GPU时间: {gpu_time:.4f}s")
    print(f"  ✓ 加速比: {speedup:.2f}x")


def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print("Taylor模型单元测试")
    print("="*70)
    
    tests = [
        test_taylor_model_creation,
        test_taylor_arithmetic,
        test_bernstein_approximation,
        test_bounds_computation,
        test_gpu_acceleration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"测试结果: {passed}通过, {failed}失败")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)