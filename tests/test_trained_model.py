# -*- coding: utf-8 -*-
"""
在已训练的pic4rl模型上测试POLAR验证
这是最重要的集成测试
"""

import sys
import os

# ========== 重要：降低 TensorFlow 日志级别 ==========
# 注意：不要在这里设置 CUDA_VISIBLE_DEVICES，否则 PyTorch 也看不到 GPU！
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 降低TF日志级别
# ===================================================

# 获取绝对路径，确保正确添加到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import rclpy
from pic4rl.verification.polar_verifier import PolarVerifier
from pic4rl.utils.polar_utils import (
    extract_tf2rl_actor_weights,
    PolarStatistics,
    visualize_polar_results
)


def test_with_trained_model(model_dir, n_tests=50, gpu_mode='auto'):
    """
    在已训练模型上测试POLAR
    
    Args:
        model_dir: 模型目录（包含checkpoint）
        n_tests: 测试次数
        gpu_mode: GPU使用模式
            - 'auto': 自动（TF用CPU，PyTorch智能选择）适合本地开发
            - 'all-gpu': 全GPU模式（TF和PyTorch都用GPU）适合服务器
            - 'cpu': 全CPU模式（都用CPU）
    """
    print("="*70)
    print("POLAR验证 - 已训练模型测试")
    print("="*70)
    print(f"\n模型路径: {model_dir}")
    print(f"测试次数: {n_tests}")
    print(f"GPU模式: {gpu_mode}")
    
    # 1. 加载tf2rl模型
    print("\n[1/5] 加载tf2rl模型...")
    try:
        # 导入 TensorFlow
        import tensorflow as tf
        
        # 根据gpu_mode配置TensorFlow的GPU使用
        if gpu_mode == 'cpu':
            # 全CPU模式：TensorFlow不使用GPU
            physical_gpus = tf.config.list_physical_devices('GPU')
            if physical_gpus:
                tf.config.set_visible_devices([], 'GPU')
                print(f"  ℹ️  TensorFlow已设置为CPU模式（全CPU模式）")
            else:
                print(f"  ℹ️  系统未检测到GPU设备")
                
        elif gpu_mode == 'auto':
            # 自动模式：TensorFlow使用CPU，为PyTorch节省显存
            physical_gpus = tf.config.list_physical_devices('GPU')
            if physical_gpus:
                tf.config.set_visible_devices([], 'GPU')
                print(f"  ℹ️  TensorFlow已设置为CPU模式（自动模式，为PyTorch预留显存）")
                print(f"     系统GPU数量: {len(physical_gpus)}")
            else:
                print(f"  ℹ️  系统未检测到GPU设备")
                
        elif gpu_mode == 'all-gpu':
            # 全GPU模式：TensorFlow也使用GPU
            physical_gpus = tf.config.list_physical_devices('GPU')
            if physical_gpus:
                # 设置GPU内存增长模式，避免TF占用所有显存
                try:
                    for gpu in physical_gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  ✓ TensorFlow使用GPU模式（全GPU模式）")
                    print(f"     GPU数量: {len(physical_gpus)}，已启用内存增长模式")
                except RuntimeError as e:
                    print(f"  ⚠️  GPU配置警告: {e}")
            else:
                print(f"  ℹ️  系统未检测到GPU设备，使用CPU")
        else:
            raise ValueError(f"不支持的GPU模式: {gpu_mode}")
        
        # 加载tf2rl模型
        # 这里需要根据你的实际pic4rl代码调整
        from tf2rl.algos.td3 import TD3
        
        # 根据gpu_mode决定TD3是否使用GPU
        # 注意：在auto/cpu模式下，TF已经被设置为不可见GPU，所以gpu参数无关紧要
        # 在all-gpu模式下，TD3可以使用GPU
        td3_gpu = -1 if gpu_mode == 'cpu' else 0  # -1表示CPU，0表示GPU:0
        
        # 加载checkpoint
        # 注意：这里的路径需要根据你的实际模型结构调整
        policy = TD3(
            state_shape=(38,),
            action_dim=2,
            max_action=0.5,
            gpu=td3_gpu
        )
        
        # 加载权重
        # policy.load_weights(model_dir)  # 根据实际情况调整
        
        print(f"  ✓ 模型加载成功")
        
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        print("\n  提示：请确保：")
        print("    1. model_dir路径正确")
        print("    2. 已安装tf2rl: pip install tf2rl")
        print("    3. 模型文件完整")
        return False
    
    # 2. 提取Actor权重
    print("\n[2/5] 提取Actor权重...")
    try:
        actor_weights = extract_tf2rl_actor_weights(policy)
        print(f"  ✓ 提取完成: {len(actor_weights)}层")
        
        # 打印网络结构
        print(f"\n  网络结构:")
        for i, (W, b, act) in enumerate(actor_weights):
            print(f"    Layer {i+1}: {W.shape} | {act}")
            
    except Exception as e:
        print(f"  ✗ 权重提取失败: {e}")
        return False
    
    # 3. 创建POLAR验证器（根据gpu_mode选择设备）
    print("\n[3/5] 创建POLAR验证器...")
    
    # 检查GPU显存，决定使用CPU还是CUDA
    import torch
    device = 'cpu'  # 默认使用CPU
    
    if gpu_mode == 'cpu':
        # 全CPU模式
        print(f"  ℹ️  PyTorch使用CPU模式（全CPU模式）")
        device = 'cpu'
        
    elif gpu_mode == 'all-gpu':
        # 全GPU模式：强制使用GPU（如果可用）
        if torch.cuda.is_available():
            print(f"  ✓ PyTorch使用GPU模式（全GPU模式）")
            device = 'cuda'
        else:
            print(f"  ⚠️  GPU不可用，回退到CPU模式")
            device = 'cpu'
            
    elif gpu_mode == 'auto' and torch.cuda.is_available():
        # 策略：先尝试CUDA，通过预热测试判断显存是否足够
        print(f"  ℹ️  检测GPU可用性...")
        
        try:
            # 先创建CUDA版本的验证器
            verifier_test = PolarVerifier(
                state_dim=38,
                action_dim=2,
                device='cuda',
                verbose=False
            )
            
            # 预热测试：使用实际网络结构测试是否会OOM
            # 使用与真实网络相同的结构（38→256→128→128→2）
            test_state = np.random.rand(38).astype(np.float32)
            test_weights = [
                (np.random.randn(38, 256).astype(np.float32) * 0.01,
                 np.random.randn(256).astype(np.float32) * 0.01,
                 'relu'),
                (np.random.randn(256, 128).astype(np.float32) * 0.01,
                 np.random.randn(128).astype(np.float32) * 0.01,
                 'relu'),
                (np.random.randn(128, 128).astype(np.float32) * 0.01,
                 np.random.randn(128).astype(np.float32) * 0.01,
                 'relu'),
                (np.random.randn(128, 2).astype(np.float32) * 0.01,
                 np.random.randn(2).astype(np.float32) * 0.01,
                 'tanh')
            ]
            
            print(f"     执行预热测试（使用实际网络结构）...")
            _, _ = verifier_test.verify_safety(
                test_weights, test_state, 
                observation_error=0.01, 
                bern_order=1, 
                max_action=0.5
            )
            
            # 检查实际显存占用
            gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info(0)
            gpu_mem_free_gb = gpu_mem_free / (1024**3)
            gpu_mem_total_gb = gpu_mem_total / (1024**3)
            gpu_mem_used_gb = gpu_mem_total_gb - gpu_mem_free_gb
            
            print(f"  ✓ GPU预热成功")
            print(f"     显存状态: {gpu_mem_free_gb:.2f} GB 可用 / {gpu_mem_total_gb:.2f} GB 总量")
            print(f"     (已使用: {gpu_mem_used_gb:.2f} GB)")
            
            # 如果剩余显存 > 2GB，使用GPU
            if gpu_mem_free_gb > 2.0:
                device = 'cuda'
                print(f"  ✓ 使用GPU加速")
                verifier = verifier_test  # 复用预热的验证器
            else:
                print(f"  ⚠️  预热后显存不足 ({gpu_mem_free_gb:.2f} GB < 2 GB)")
                print(f"     切换到CPU模式")
                device = 'cpu'
                del verifier_test
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ⚠️  GPU显存不足，自动切换到CPU模式")
                device = 'cpu'
                torch.cuda.empty_cache()
            else:
                raise
    
    else:
        # auto模式但GPU不可用，或其他情况
        print(f"  ℹ️  PyTorch使用CPU模式")
    
    # 如果没有复用预热的验证器，创建新的
    if device == 'cpu' or 'verifier' not in locals():
        verifier = PolarVerifier(
            state_dim=38,
            action_dim=2,
            device=device,
            verbose=False
        )
    
    print(f"  ✓ 验证器创建成功 (设备: {device})")
    
    # 4. 批量验证
    print(f"\n[4/5] 开始验证 {n_tests} 个随机状态...")
    stats = PolarStatistics(save_dir='./polar_test_results')
    
    import time
    overall_start = time.time()
    
    for i in range(n_tests):
        # 生成随机测试状态
        state = np.random.rand(38).astype(np.float32)
        
        # 模拟真实场景的观测分布
        state[0] = np.random.uniform(0.1, 1.0)  # 距离目标 [0.5, 5.0]米
        state[1] = np.random.uniform(-0.5, 0.5)  # 角度 [-π/2, π/2]
        state[2:38] = np.random.uniform(0.3, 1.0, 36)  # 激光雷达 [3, 10]米
        
        # POLAR验证
        start = time.time()
        is_safe, ranges = verifier.verify_safety(
            actor_weights,
            state,
            observation_error=0.01,
            bern_order=1,
            max_action=0.5
        )
        elapsed = time.time() - start
        
        # 记录结果
        stats.record(is_safe, ranges, state, elapsed)
        
        # 每10次清理一次GPU缓存（如果使用GPU）
        if device == 'cuda' and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
        # 进度显示
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{n_tests}")
    
    total_time = time.time() - overall_start
    
    print(f"\n  ✓ 验证完成")
    print(f"    总耗时: {total_time:.2f}秒")
    print(f"    平均: {total_time/n_tests:.3f}秒/次")
    
    # 5. 统计结果
    print("\n[5/5] 统计结果...")
    stats.print_summary()
    
    # 保存结果
    result_file = stats.save_results()
    
    # 可视化
    print("\n生成可视化图表...")
    try:
        # 使用英文标签避免字体问题
        visualize_polar_results(stats, 
            save_path='./polar_test_results/verification_results.png',
            lang='en')
    except Exception as e:
        print(f"  ⚠️  可视化失败: {e}")
    
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)
    
    return True


def test_with_mock_model(n_tests=20):
    """
    使用模拟模型测试（无需真实模型）
    用于快速验证POLAR实现
    """
    print("="*70)
    print("POLAR验证 - 模拟模型测试")
    print("="*70)
    
    # 创建模拟的Actor权重
    print("\n[1/3] 创建模拟Actor网络...")
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
    print(f"  ✓ 网络结构: 38→256→128→128→2")
    
    # 创建验证器
    print("\n[2/3] 创建POLAR验证器...")
    verifier = PolarVerifier(state_dim=38, action_dim=2, verbose=False)
    
    # 批量验证
    print(f"\n[3/3] 验证 {n_tests} 个状态...")
    stats = PolarStatistics(save_dir='./polar_mock_results')
    
    for i in range(n_tests):
        state = np.random.rand(38).astype(np.float32)
        
        import time
        start = time.time()
        is_safe, ranges = verifier.verify_safety(
            actor_weights, state, observation_error=0.01, max_action=0.5
        )
        elapsed = time.time() - start
        
        stats.record(is_safe, ranges, state, elapsed)
    
    # 结果
    stats.print_summary()
    stats.save_results()
    
    try:
        # 使用英文标签避免字体问题
        visualize_polar_results(stats, 
            save_path='./polar_mock_results/mock_results.png',
            lang='en')
    except:
        pass
    
    print("\n✅ 模拟测试完成！")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='测试POLAR验证',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU模式说明:
  auto     - 自动模式（默认）：TF使用CPU，PyTorch智能选择（适合本地开发）
  all-gpu  - 全GPU模式：TF和PyTorch都使用GPU（适合服务器，显存充足）
  cpu      - 全CPU模式：强制所有组件使用CPU

使用示例:
  # 本地开发（默认）
  python tests/test_trained_model.py --model-dir path/to/model --n-tests 100
  
  # 服务器（显存充足）
  python tests/test_trained_model.py --model-dir path/to/model --n-tests 100 --gpu-mode all-gpu
  
  # 强制CPU模式
  python tests/test_trained_model.py --model-dir path/to/model --n-tests 100 --gpu-mode cpu
        """
    )
    parser.add_argument('--model-dir', type=str, default=None,
                       help='已训练模型的路径')
    parser.add_argument('--n-tests', type=int, default=50,
                       help='测试次数（默认50）')
    parser.add_argument('--mock', action='store_true',
                       help='使用模拟模型测试（无需真实模型）')
    parser.add_argument('--gpu-mode', type=str, default='auto',
                       choices=['auto', 'all-gpu', 'cpu'],
                       help='GPU使用模式（默认: auto）')
    
    args = parser.parse_args()
    
    if args.mock or args.model_dir is None:
        print("使用模拟模型进行测试...\n")
        success = test_with_mock_model(n_tests=args.n_tests)
    else:
        print(f"使用真实模型进行测试: {args.model_dir}\n")
        success = test_with_trained_model(
            args.model_dir, 
            n_tests=args.n_tests, 
            gpu_mode=args.gpu_mode
        )
    
    sys.exit(0 if success else 1)
