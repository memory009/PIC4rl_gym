"""
POLAR辅助工具
1. 从tf2rl提取Actor权重
2. 统计记录和可视化
3. 结果保存
"""

import numpy as np
import os
import json
import pickle
from datetime import datetime


def extract_tf2rl_actor_weights(policy):
    """
    从tf2rl的TD3 policy中提取Actor权重
    
    Args:
        policy: tf2rl TD3 policy对象
    
    Returns:
        weights_list: [
            (W1, b1, 'relu'),
            (W2, b2, 'relu'),
            (W3, b3, 'tanh'),
            ...
        ]
    """
    try:
        import tensorflow as tf
        # 确保 TensorFlow 不使用 GPU（避免与 PyTorch 的 POLAR 验证冲突）
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
    except ImportError:
        raise ImportError("需要安装TensorFlow: pip install tensorflow")
    
    # 获取Actor模型
    actor_model = policy.actor
    
    weights_list = []
    
    # 遍历所有层
    for layer in actor_model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            continue
        
        # 提取权重和偏置
        layer_weights = layer.get_weights()
        W = layer_weights[0]  # shape: [in_dim, out_dim]
        b = layer_weights[1]  # shape: [out_dim]
        
        # 判断激活函数
        activation_fn = layer.activation
        
        if activation_fn == tf.keras.activations.relu:
            activation_name = 'relu'
        elif activation_fn == tf.keras.activations.tanh:
            activation_name = 'tanh'
        elif activation_fn == tf.keras.activations.linear:
            activation_name = 'linear'
        else:
            activation_name = 'unknown'
            print(f"⚠️  未知激活函数: {activation_fn}")
        
        weights_list.append((
            W.astype(np.float32),
            b.astype(np.float32),
            activation_name
        ))
        
        print(f"  Layer {len(weights_list)}: {W.shape} | {activation_name}")
    
    return weights_list


def save_polar_weights(weights_list, save_path):
    """
    保存提取的权重（用于POLAR验证）
    
    Args:
        weights_list: 权重列表
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(weights_list, f)
    
    print(f"✓ 权重已保存: {save_path}")


def load_polar_weights(load_path):
    """加载POLAR权重"""
    with open(load_path, 'rb') as f:
        weights_list = pickle.load(f)
    
    print(f"✓ 权重已加载: {load_path}")
    return weights_list


class PolarStatistics:
    """POLAR验证统计记录器"""
    
    def __init__(self, save_dir=None):
        self.save_dir = save_dir or './polar_results'
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.records = []
        self.safe_count = 0
        self.unsafe_count = 0
        self.error_count = 0
        
        self.v_ranges = []
        self.w_ranges = []
        
        self.computation_times = []
    
    def record(self, is_safe, ranges, state=None, computation_time=None):
        """
        记录一次验证结果
        
        Args:
            is_safe: bool
            ranges: [[v_min, v_max], [w_min, w_max]]
            state: 观测状态（可选）
            computation_time: 计算时间（可选）
        """
        if is_safe:
            self.safe_count += 1
        else:
            self.unsafe_count += 1
        
        # 记录可达集宽度
        v_width = ranges[0][1] - ranges[0][0]
        w_width = ranges[1][1] - ranges[1][0]
        
        self.v_ranges.append(v_width)
        self.w_ranges.append(w_width)
        
        if computation_time is not None:
            self.computation_times.append(computation_time)
        
        # 详细记录
        record = {
            'timestamp': datetime.now().isoformat(),
            'is_safe': is_safe,
            'ranges': ranges,
            'v_width': v_width,
            'w_width': w_width,
            'computation_time': computation_time
        }
        
        if state is not None:
            record['state'] = {
                'distance': float(state[0]),
                'bearing': float(state[1]),
                'min_laser': float(np.min(state[2:38])) if len(state) >= 38 else None
            }
        
        self.records.append(record)
    
    def record_error(self, error_msg):
        """记录验证错误"""
        self.error_count += 1
        self.records.append({
            'timestamp': datetime.now().isoformat(),
            'error': error_msg
        })
    
    def get_summary(self):
        """获取统计摘要"""
        total = self.safe_count + self.unsafe_count
        
        summary = {
            'total_verifications': total,
            'safe_count': self.safe_count,
            'unsafe_count': self.unsafe_count,
            'error_count': self.error_count,
            'safety_rate': self.safe_count / total if total > 0 else 0.0,
            
            'avg_v_width': np.mean(self.v_ranges) if self.v_ranges else 0.0,
            'avg_w_width': np.mean(self.w_ranges) if self.w_ranges else 0.0,
            'max_v_width': np.max(self.v_ranges) if self.v_ranges else 0.0,
            'max_w_width': np.max(self.w_ranges) if self.w_ranges else 0.0,
            
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0.0
        }
        
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("POLAR验证统计摘要")
        print("="*70)
        print(f"  总验证次数: {summary['total_verifications']}")
        print(f"  安全状态:   {summary['safe_count']} ({summary['safety_rate']:.2%})")
        print(f"  不安全状态: {summary['unsafe_count']}")
        print(f"  验证错误:   {summary['error_count']}")
        print()
        print(f"  可达集宽度统计:")
        print(f"    线速度: 平均={summary['avg_v_width']:.4f}, 最大={summary['max_v_width']:.4f}")
        print(f"    角速度: 平均={summary['avg_w_width']:.4f}, 最大={summary['max_w_width']:.4f}")
        print()
        print(f"  平均计算时间: {summary['avg_computation_time']:.3f}秒")
        print("="*70)
    
    def save_results(self, filename=None):
        """保存结果到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"polar_results_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        results = {
            'summary': self.get_summary(),
            'records': self.records
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ 结果已保存: {filepath}")
        return filepath


def visualize_polar_results(stats, save_path=None, lang='auto'):
    """
    可视化POLAR验证结果
    
    Args:
        stats: PolarStatistics对象
        save_path: 图片保存路径
        lang: 'auto', 'zh', 'en' - 语言选择
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError:
        print("⚠️  需要安装matplotlib: pip install matplotlib")
        return
    
    if not stats.v_ranges:
        print("⚠️  没有数据可视化")
        return
    
    # 配置中文字体
    use_chinese = False
    if lang in ['auto', 'zh']:
        try:
            import matplotlib.font_manager as fm
            
            # 测试字符串，包含常见的中文字符
            test_chars = '安全错误不验证'
            
            # 尝试配置中文字体（按优先级）
            chinese_fonts = [
                'Noto Sans CJK SC',     # Linux Noto CJK 简体中文
                'Noto Sans CJK TC',     # Linux Noto CJK 繁体中文
                'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
                'WenQuanYi Zen Hei',    # Linux 文泉驿正黑
                'SimHei',               # Windows 黑体
                'Microsoft YaHei',      # Windows 微软雅黑
                'PingFang SC',          # macOS 苹方
                'Heiti SC',             # macOS 黑体-简
                'STHeiti',              # macOS 华文黑体
            ]
            
            # 获取系统可用的字体
            available_fonts = {f.name: f for f in fm.fontManager.ttflist}
            
            for font_name in chinese_fonts:
                if font_name in available_fonts:
                    try:
                        # 配置字体
                        matplotlib.rcParams['font.sans-serif'] = [font_name]
                        matplotlib.rcParams['font.family'] = 'sans-serif'
                        matplotlib.rcParams['axes.unicode_minus'] = False
                        
                        # 验证字体是否真的支持中文
                        # 通过检查字体文件是否包含CJK字符
                        font_entry = available_fonts[font_name]
                        if 'CJK' in font_entry.name or 'Hei' in font_entry.name or font_name in ['SimHei', 'Microsoft YaHei']:
                            use_chinese = True
                            break
                    except:
                        continue
            
            # 如果没有找到合适的字体，强制使用英文
            if not use_chinese and lang == 'zh':
                print(f"⚠️  未找到完整的中文字体，使用英文标签")
                
        except Exception as e:
            pass
    
    # 根据字体支持选择标签
    if use_chinese and lang != 'en':
        labels_dict = {
            'safe': '安全', 'unsafe': '不安全', 'error': '错误',
            'safety_dist': '安全性分布',
            'linear_vel': '线速度', 'angular_vel': '角速度',
            'reachable_width': '可达集宽度', 'frequency': '频数',
            'width_dist': '可达集宽度分布',
            'linear_width': '线速度宽度', 'angular_width': '角速度宽度',
            'verification_idx': '验证序号', 'width_evolution': '可达集宽度演化',
            'computation_time': '计算时间 (秒)', 'time_dist': '验证计算时间分布'
        }
    else:
        labels_dict = {
            'safe': 'Safe', 'unsafe': 'Unsafe', 'error': 'Error',
            'safety_dist': 'Safety Distribution',
            'linear_vel': 'Linear Velocity', 'angular_vel': 'Angular Velocity',
            'reachable_width': 'Reachable Set Width', 'frequency': 'Frequency',
            'width_dist': 'Reachable Set Width Distribution',
            'linear_width': 'Linear Vel Width', 'angular_width': 'Angular Vel Width',
            'verification_idx': 'Verification Index', 'width_evolution': 'Width Evolution',
            'computation_time': 'Computation Time (s)', 'time_dist': 'Computation Time Distribution'
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 安全率饼图 - 使用简单版本，强制英文标签避免字体问题
    ax1 = axes[0, 0]
    # 强制使用英文标签
    pie_labels = ['Safe', 'Unsafe', 'Error']
    sizes = [stats.safe_count, stats.unsafe_count, stats.error_count]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    # 检测是否有极端比例（>95%），避免标签与标题重叠
    total = sum(sizes)
    max_ratio = max(sizes) / total if total > 0 else 0
    
    if max_ratio > 0.95:
        # 极端情况：使用图例代替直接标注
        wedges, texts, autotexts = ax1.pie(sizes, colors=colors, autopct='%1.1f%%', 
                                            startangle=90, labels=None)
        ax1.legend(wedges, pie_labels, loc='center left', bbox_to_anchor=(0.85, 0, 0.5, 1))
    else:
        # 正常情况：直接标注
        ax1.pie(sizes, labels=pie_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    
    ax1.set_title('Safety Distribution')
    
    # 2. 可达集宽度分布 - 使用英文标签
    ax2 = axes[0, 1]
    ax2.hist(stats.v_ranges, bins=30, alpha=0.7, label='Linear Velocity', color='blue')
    ax2.hist(stats.w_ranges, bins=30, alpha=0.7, label='Angular Velocity', color='red')
    ax2.set_xlabel('Reachable Set Width')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reachable Set Width Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 可达集宽度趋势 - 使用英文标签
    ax3 = axes[1, 0]
    ax3.plot(stats.v_ranges, label='Linear Vel Width', alpha=0.7)
    ax3.plot(stats.w_ranges, label='Angular Vel Width', alpha=0.7)
    ax3.set_xlabel('Verification Index')
    ax3.set_ylabel('Reachable Set Width')
    ax3.set_title('Width Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 计算时间分布 - 使用英文标签
    ax4 = axes[1, 1]
    if stats.computation_times:
        ax4.hist(stats.computation_times, bins=30, color='green', alpha=0.7)
        ax4.set_xlabel('Computation Time (s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Computation Time Distribution')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {save_path}")
    
    plt.show()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("="*70)
    print("POLAR工具测试")
    print("="*70)
    
    # 测试统计记录器
    print("\n[1/2] 测试统计记录器...")
    stats = PolarStatistics(save_dir='./test_polar_results')
    
    # 模拟一些验证结果
    for i in range(50):
        is_safe = np.random.random() > 0.2
        v_range = [0.0, 0.1 + np.random.random() * 0.4]
        w_range = [-0.5, 0.5 + np.random.random() * 0.3]
        ranges = [v_range, w_range]
        
        state = np.random.rand(38).astype(np.float32)
        computation_time = 0.5 + np.random.random() * 2.0
        
        stats.record(is_safe, ranges, state, computation_time)
    
    # 模拟一些错误
    for i in range(3):
        stats.record_error(f"测试错误 {i+1}")
    
    print("  ✓ 记录了53次验证结果")
    
    # 打印摘要
    stats.print_summary()
    
    # 保存结果
    result_file = stats.save_results('test_results.json')
    
    # 可视化
    print("\n[2/2] 生成可视化图表...")
    visualize_polar_results(stats, save_path='./test_polar_results/visualization.png')
    
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)
    
    # 清理测试文件
    import shutil
    if os.path.exists('./test_polar_results'):
        shutil.rmtree('./test_polar_results')
        print("\n✓ 测试文件已清理")