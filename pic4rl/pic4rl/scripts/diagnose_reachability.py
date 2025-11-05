#!/usr/bin/env python3
"""
POLAR可达性诊断脚本

用法:
    python3 scripts/diagnose_reachability.py

功能:
    1. 加载TD3模型权重
    2. 定义测试场景
    3. 计算动作可达集
    4. 生成安全性报告
"""

import numpy as np
import json
import os
import sys
from pathlib import Path

# 添加项目路径 - 修正：需要到pic4rl包的父目录
script_dir = Path(__file__).resolve().parent  # scripts/
pic4rl_package = script_dir.parent  # pic4rl/
project_root = pic4rl_package.parent  # PIC4rl_gym/pic4rl/
sys.path.insert(0, str(project_root))

from pic4rl.verification import (
    TD3WeightExtractor,
    POLARReachability,
    analyze_safety
)


def define_test_scenarios():
    """
    定义测试场景
    
    场景设计原则:
    1. 正常导航（远离障碍物）
    2. 接近障碍物（安全距离）
    3. 危险场景（非常接近障碍物）
    4. 目标接近（即将到达）
    """
    scenarios = {
        'scenario1_normal': {
            'description': '正常导航：目标在前方5m，无近距离障碍物',
            'state': np.array(
                [5.0, 0.0] + [10.0] * 36,  # 距离5m, 角度0, 激光全是10m
                dtype=np.float32
            ),
            'observation_error': 0.01
        },
        
        'scenario2_turning': {
            'description': '需要转向：目标在右侧45度，3m远',
            'state': np.array(
                [3.0, 0.785] + [10.0] * 36,  # 距离3m, 角度45度(0.785rad)
                dtype=np.float32
            ),
            'observation_error': 0.01
        },
        
        'scenario3_obstacle_ahead': {
            'description': '前方有障碍物：目标5m远，但前方1m处有障碍',
            'state': np.array(
                [5.0, 0.0] + [1.0, 1.0, 1.0] + [10.0] * 33,  # 前3个激光点1m
                dtype=np.float32
            ),
            'observation_error': 0.01
        },
        
        'scenario4_near_goal': {
            'description': '接近目标：目标在0.5m远，前方无障碍',
            'state': np.array(
                [0.5, 0.1] + [10.0] * 36,  # 距离0.5m, 角度微偏
                dtype=np.float32
            ),
            'observation_error': 0.01
        },
        
        'scenario5_dangerous': {
            'description': '危险场景：目标3m远，但周围都是障碍物',
            'state': np.array(
                [3.0, 0.2] + [0.3] * 36,  # 周围全是0.3m的障碍
                dtype=np.float32
            ),
            'observation_error': 0.01
        },
    }
    
    return scenarios


def run_diagnosis(checkpoint_dir, output_dir='reachability_results'):
    """
    运行完整的可达性诊断
    
    Args:
        checkpoint_dir: TD3模型checkpoint目录
        output_dir: 结果输出目录
    """
    print("="*70)
    print("POLAR可达性诊断工具")
    print("="*70)
    
    # Step 1: 加载模型权重
    print("\n[1/4] 加载TD3模型权重...")
    extractor = TD3WeightExtractor(checkpoint_dir)
    weights, biases = extractor.load_actor_weights()
    
    # Step 2: 初始化POLAR验证器
    print("\n[2/4] 初始化POLAR验证器...")
    polar = POLARReachability(
        weights=weights,
        biases=biases,
        max_action=0.5,  # 你的TD3配置
        bern_order=1,
        error_steps=4000
    )
    
    # Step 3: 运行测试场景
    print("\n[3/4] 运行测试场景...")
    scenarios = define_test_scenarios()
    results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        print("\n" + "="*70)
        print(f"场景: {scenario_name}")
        print(f"描述: {scenario_data['description']}")
        print("="*70)
        
        # 计算可达集
        action_ranges, comp_time = polar.compute_reachable_set(
            state=scenario_data['state'],
            observation_error=scenario_data['observation_error']
        )
        
        # 安全性分析
        is_safe, reasons = analyze_safety(action_ranges, scenario_data['state'])
        
        # 保存结果
        results[scenario_name] = {
            'description': scenario_data['description'],
            'state': scenario_data['state'].tolist(),
            'observation_error': scenario_data['observation_error'],
            'action_ranges': {
                'linear_velocity': action_ranges[0],
                'angular_velocity': action_ranges[1]
            },
            'computation_time': comp_time,
            'safety': {
                'is_safe': is_safe,
                'reasons': reasons
            }
        }
        
        # 打印安全性分析
        print(f"\n{'='*70}")
        print("安全性分析:")
        if is_safe:
            print("  ✅ 场景安全")
        else:
            print("  ❌ 场景不安全")
            for reason in reasons:
                print(f"    - {reason}")
    
    # Step 4: 保存报告
    print("\n" + "="*70)
    print("[4/4] 生成诊断报告...")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存JSON报告
    report_path = os.path.join(output_dir, 'reachability_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ JSON报告已保存: {report_path}")
    
    # 生成文本摘要
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("POLAR可达性诊断摘要\n")
        f.write("="*70 + "\n\n")
        
        for scenario_name, result in results.items():
            f.write(f"场景: {scenario_name}\n")
            f.write(f"描述: {result['description']}\n")
            f.write(f"线速度范围: [{result['action_ranges']['linear_velocity'][0]:.6f}, "
                   f"{result['action_ranges']['linear_velocity'][1]:.6f}] m/s\n")
            f.write(f"角速度范围: [{result['action_ranges']['angular_velocity'][0]:.6f}, "
                   f"{result['action_ranges']['angular_velocity'][1]:.6f}] rad/s\n")
            f.write(f"计算时间: {result['computation_time']:.3f}s\n")
            f.write(f"安全性: {'✅ 安全' if result['safety']['is_safe'] else '❌ 不安全'}\n")
            if not result['safety']['is_safe']:
                for reason in result['safety']['reasons']:
                    f.write(f"  - {reason}\n")
            f.write("\n" + "-"*70 + "\n\n")
    
    print(f"✅ 文本摘要已保存: {summary_path}")
    
    # 打印最终统计
    safe_count = sum(1 for r in results.values() if r['safety']['is_safe'])
    total_count = len(results)
    
    print("\n" + "="*70)
    print("诊断完成！")
    print(f"总场景数: {total_count}")
    print(f"安全场景: {safe_count}")
    print(f"不安全场景: {total_count - safe_count}")
    print(f"安全率: {safe_count/total_count*100:.1f}%")
    print("="*70)


if __name__ == '__main__':
    # 配置
    checkpoint_dir = "/home/cheeson/pic4rl_ws/src/Results/20251031_120540.356112_lidar_TD3/20251031T120540.688820_TD3_"
    output_dir = "reachability_results"
    
    # 运行诊断
    run_diagnosis(checkpoint_dir, output_dir)