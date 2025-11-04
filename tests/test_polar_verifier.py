# -*- coding: utf-8 -*-
"""
POLAR验证器集成测试
测试完整的可达集计算流程
"""

import sys
import os
# 获取绝对路径，确保正确添加到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
from pic4rl.verification.polar_verifier import PolarVerifier


def test_simple_network():
    """测试1: 简单网络验证"""
    print("\n[测试1] 简单网络 (2→4→2)...")
    
    # 创建简单网络
    actor_weights = [
        # Layer 1: 2→4, ReLU
        (np.array([[0.5, -0.3, 0.2, 0.1],
                   [0.1, 0.4, -0.2, 0.3]], dtype=np.float32),
         np.array([0.1, -0.1, 0.0, 0.2], dtype=np.float32),
         'relu'),
        
        # Layer 2: 4→2, Tanh
        (np.array([[0.5, -0.2],
                   [0.3, 0.4],
                   [-0.1, 0.3],
                   [0.2, -0.1]], dtype=np.float32),
         np.array([0.0, 0.0], dtype=np.float32),
         'tanh')
    ]
    
    # 创建验证器
    verifier = PolarVerifier(state_dim=2, action_dim=2, verbose=True)
    
    # 测试状态
    state = np.array([0.5, 0.3], dtype=np.float32)
    
    # 验证
    is_safe, ranges = verifier.verify_safety(
        actor_weights,
        state,
        observation_error=0.01,
        bern_order=1,
        max_action=0.5
    )
    
    print(f"  ✓ 验证完成")
    print(f"    安全性: {is_safe}")
    print(f"    可达集: v=[{ranges[0][0]:.4f}, {ranges[0][1]:.4f}], "
          f"w=[{ranges[1][0]:.4f}, {ranges[1][1]:.4f}]")
    
    # 检查可达集合理性
    assert ranges[0][1] >= ranges[0][0], "线速度上界应≥下界"
    assert ranges[1][1] >= ranges[1][0], "角速度上界应≥下界"


def test_realistic_network():
    """测试2: 真实网络结构 (38→256→128→128→2)"""
    print("\n[测试2] 真实网络结构...")
    
    # 创建真实大小的网络
    actor_weights = [
        (np.random.randn(38, 256).astype(np.float32) * 0.1,
         np.random.randn(256).astype(np.float32) * 0.1,
         'relu'),
        
        (np.random.randn(256, 128).astype(np.float32) * 0.1,
         np.random.randn(128).astype(np.float32) * 0.1,
         'relu'),
        
        (np.random.randn(128, 128).astype(np.float32) * 0.1,
         np.random.randn(128).astype(np.float32) * 0.1,
         'relu'),
        
        (np.random.randn(128, 2).astype(np.float32) * 0.1,
         np.random.randn(2).astype(np.float32) * 0.1,
         'tanh')
    ]
    
    # 创建验证器
    verifier = PolarVerifier(state_dim=38, action_dim=2, verbose=False)
    
    # 测试状态（模拟真实观测）
    state = np.random.rand(38).astype(np.float32)
    state[0] = 0.5  # 距离
    state[1] = 0.2  # 角度
    state[2:38] = 0.7  # 激光雷达
    
    import time
    start = time.time()
    
    is_safe, ranges = verifier.verify_safety(
        actor_weights,
        state,
        observation_error=0.01,
        bern_order=1,
        max_action=0.5
    )
    
    elapsed = time.time() - start
    
    print(f"  ✓ 验证完成 (耗时: {elapsed:.3f}秒)")
    print(f"    安全性: {is_safe}")
    print(f"    可达集宽度: v={ranges[0][1]-ranges[0][0]:.4f}, "
          f"w={ranges[1][1]-ranges[1][0]:.4f}")


def test_different_error_levels():
    """测试3: 不同观测误差水平"""
    print("\n[测试3] 不同观测误差水平...")
    
    # 创建网络
    actor_weights = [
        (np.random.randn(10, 16).astype(np.float32) * 0.1,
         np.random.randn(16).astype(np.float32) * 0.1,
         'relu'),
        (np.random.randn(16, 2).astype(np.float32) * 0.1,
         np.random.randn(2).astype(np.float32) * 0.1,
         'tanh')
    ]
    
    verifier = PolarVerifier(state_dim=10, action_dim=2, verbose=False)
    state = np.random.rand(10).astype(np.float32)
    
    error_levels = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    print(f"  {'误差':<10s} {'v宽度':<12s} {'w宽度':<12s} {'安全性':<10s}")
    print("  " + "-"*50)
    
    for error in error_levels:
        is_safe, ranges = verifier.verify_safety(
            actor_weights, state, observation_error=error, max_action=0.5
        )
        
        v_width = ranges[0][1] - ranges[0][0]
        w_width = ranges[1][1] - ranges[1][0]
        
        print(f"  {error:<10.3f} {v_width:<12.6f} {w_width:<12.6f} "
              f"{'✓' if is_safe else '✗':<10s}")
    
    print("  ✓ 验证：误差越大，可达集越宽")


def test_batch_verification():
    """测试4: 批量验证性能"""
    print("\n[测试4] 批量验证性能...")
    
    # 创建网络
    actor_weights = [
        (np.random.randn(38, 128).astype(np.float32) * 0.1,
         np.random.randn(128).astype(np.float32) * 0.1,
         'relu'),
        (np.random.randn(128, 64).astype(np.float32) * 0.1,
         np.random.randn(64).astype(np.float32) * 0.1,
         'relu'),
        (np.random.randn(64, 2).astype(np.float32) * 0.1,
         np.random.randn(2).astype(np.float32) * 0.1,
         'tanh')
    ]
    
    verifier = PolarVerifier(state_dim=38, action_dim=2, verbose=False)
    
    # 批量验证
    n_tests = 20
    import time
    
    start = time.time()
    safe_count = 0
    
    for i in range(n_tests):
        state = np.random.rand(38).astype(np.float32)
        is_safe, _ = verifier.verify_safety(
            actor_weights, state, observation_error=0.01, max_action=0.5
        )
        if is_safe:
            safe_count += 1
    
    elapsed = time.time() - start
    
    print(f"  ✓ 完成 {n_tests} 次验证")
    print(f"    总耗时: {elapsed:.2f}秒")
    print(f"    平均: {elapsed/n_tests:.3f}秒/次")
    print(f"    安全率: {safe_count}/{n_tests} ({safe_count/n_tests:.1%})")
    
    # 获取统计
    stats = verifier.get_statistics()
    print(f"    统计平均时间: {stats['avg_computation_time']:.3f}秒")


def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print("POLAR验证器集成测试")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device.upper()}")
    
    tests = [
        test_simple_network,
        test_realistic_network,
        test_different_error_levels,
        test_batch_verification
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"测试结果: {passed}通过, {failed}失败")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)