#!/usr/bin/env python3
"""
TensorFlow模型权重提取适配器
将tf2rl的TD3模型权重转换为NumPy格式供POLAR使用
"""

import numpy as np
import tensorflow as tf
import os
import pickle
import glob


class TD3WeightExtractor:
    """
    TD3模型权重提取器
    
    兼容tf2rl的checkpoint保存格式
    """
    
    def __init__(self, checkpoint_dir):
        """
        Args:
            checkpoint_dir: checkpoint目录路径
                例如: "../Results/20251031_120540.356112_lidar_TD3/20251031T120540.688820_TD3_"
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        
        # 检查checkpoint文件
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"找不到checkpoint文件: {self.checkpoint_path}")
        
        # 读取最新的checkpoint
        with open(self.checkpoint_path, 'r') as f:
            first_line = f.readline().strip()
            # 格式: model_checkpoint_path: "ckpt-10"
            self.latest_ckpt = first_line.split('"')[1]
        
        print(f"✅ 找到checkpoint: {self.checkpoint_path}")
        print(f"✅ 最新模型: {self.latest_ckpt}")
    
    def _build_actor_network(self, state_dim=38, action_dim=2):
        """
        重建Actor网络结构（必须与训练时一致）
        
        网络结构:
            [38] → [256, ReLU] → [128, ReLU] → [128, ReLU] → [2, Tanh] → ×max_action
        """
        from tensorflow.keras import layers, Model
        
        inputs = layers.Input(shape=(state_dim,))
        
        # 隐藏层1
        x = layers.Dense(256, activation='relu', name='l1')(inputs)
        
        # 隐藏层2
        x = layers.Dense(128, activation='relu', name='l2')(x)
        
        # 隐藏层3
        x = layers.Dense(128, activation='relu', name='l3')(x)
        
        # 输出层
        outputs = layers.Dense(action_dim, activation='tanh', name='l4')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='actor')
        return model
    
    def load_actor_weights(self):
        """
        从checkpoint加载Actor网络权重
        
        Returns:
            weights: list of np.ndarray - 权重矩阵列表 [W1, W2, W3, W4]
            biases: list of np.ndarray - 偏置向量列表 [b1, b2, b3, b4]
        """
        try:
            # 1. 重建网络
            print("\n📊 重建Actor网络...")
            actor = self._build_actor_network()
            actor.summary()
            
            # 2. 创建checkpoint对象
            ckpt = tf.train.Checkpoint(actor=actor)
            
            # 3. 加载权重
            ckpt_path = os.path.join(self.checkpoint_dir, self.latest_ckpt)
            status = ckpt.restore(ckpt_path)
            
            # 尝试加载（tf2rl的保存方式）
            try:
                status.expect_partial()  # 忽略optimizer等额外变量
                print(f"✅ 成功从checkpoint加载权重: {self.latest_ckpt}")
            except:
                print(f"⚠️ 部分权重加载，尝试完整匹配...")
                status.assert_consumed()
            
            # 4. 提取权重
            weights = []
            biases = []
            
            print("\n📦 提取权重...")
            for i, layer in enumerate(actor.layers):
                if isinstance(layer, tf.keras.layers.Dense):
                    w = layer.get_weights()
                    if len(w) == 2:
                        weights.append(w[0])
                        biases.append(w[1])
                        print(f"  Layer {layer.name}: W{w[0].shape}, b{w[1].shape}")
            
            # 5. 验证网络结构
            expected_shapes = [
                ((38, 256), (256,)),   # 隐藏层1
                ((256, 128), (128,)),  # 隐藏层2
                ((128, 128), (128,)),  # 隐藏层3
                ((128, 2), (2,))       # 输出层
            ]
            
            if len(weights) != 4:
                raise ValueError(f"❌ 预期4层，实际得到{len(weights)}层")
            
            for i, ((w_shape, b_shape), (w, b)) in enumerate(zip(expected_shapes, zip(weights, biases))):
                if w.shape != w_shape:
                    raise ValueError(f"❌ 第{i+1}层权重形状错误: 预期{w_shape}, 实际{w.shape}")
                if b.shape != b_shape:
                    raise ValueError(f"❌ 第{i+1}层偏置形状错误: 预期{b_shape}, 实际{b.shape}")
            
            print(f"\n✅ 权重提取成功！共{len(weights)}层")
            return weights, biases
            
        except Exception as e:
            print(f"❌ 加载checkpoint失败，尝试直接读取变量...")
            return self._load_weights_from_checkpoint_variables()
    
    def _load_weights_from_checkpoint_variables(self):
        """
        直接从checkpoint变量读取（备用方法）
        """
        try:
            # 列出所有变量
            ckpt_path = os.path.join(self.checkpoint_dir, self.latest_ckpt)
            reader = tf.train.load_checkpoint(ckpt_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            
            print("\n📋 Checkpoint中的变量:")
            for key in sorted(var_to_shape_map.keys()):
                print(f"  {key}: {var_to_shape_map[key]}")
            
            # 提取actor相关变量
            weights = []
            biases = []
            
            for i in range(1, 5):  # l1, l2, l3, l4
                w_key = f'actor/l{i}/kernel/.ATTRIBUTES/VARIABLE_VALUE'
                b_key = f'actor/l{i}/bias/.ATTRIBUTES/VARIABLE_VALUE'
                
                if w_key in var_to_shape_map and b_key in var_to_shape_map:
                    w = reader.get_tensor(w_key)
                    b = reader.get_tensor(b_key)
                    weights.append(w)
                    biases.append(b)
                    print(f"✅ 加载 l{i}: W{w.shape}, b{b.shape}")
                else:
                    # 尝试其他可能的命名
                    for alt_w_key in [f'actor/l{i}/kernel', f'actor/_layer{i}/kernel']:
                        if alt_w_key in [k.split('/.ATTRIBUTES')[0] for k in var_to_shape_map.keys()]:
                            full_w_key = [k for k in var_to_shape_map.keys() if alt_w_key in k][0]
                            full_b_key = full_w_key.replace('kernel', 'bias')
                            
                            w = reader.get_tensor(full_w_key)
                            b = reader.get_tensor(full_b_key)
                            weights.append(w)
                            biases.append(b)
                            print(f"✅ 加载 l{i} (备用命名): W{w.shape}, b{b.shape}")
                            break
            
            if len(weights) != 4:
                raise ValueError(f"❌ 只找到{len(weights)}层权重")
            
            return weights, biases
            
        except Exception as e:
            print(f"❌ 备用方法也失败: {e}")
            raise
    
    def extract_and_save(self, output_path=None):
        """
        提取权重并保存为pickle文件
        """
        if output_path is None:
            output_path = os.path.join(self.checkpoint_dir, "actor_weights.pkl")
        
        weights, biases = self.load_actor_weights()
        
        data = {
            'weights': weights,
            'biases': biases,
            'checkpoint': self.latest_ckpt,
            'network_structure': [
                {'input': 38, 'output': 256, 'activation': 'relu'},
                {'input': 256, 'output': 128, 'activation': 'relu'},
                {'input': 128, 'output': 128, 'activation': 'relu'},
                {'input': 128, 'output': 2, 'activation': 'tanh'}
            ]
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ 权重已保存至: {output_path}")
        return output_path


def verify_weight_extraction(checkpoint_dir):
    """
    验证权重提取是否正确
    """
    print("="*70)
    print("开始验证权重提取...")
    print("="*70)
    
    try:
        extractor = TD3WeightExtractor(checkpoint_dir)
        weights, biases = extractor.load_actor_weights()
        
        # 验证1: 数值类型
        print("\n[验证1] 检查数值类型...")
        for i, (w, b) in enumerate(zip(weights, biases)):
            assert w.dtype in [np.float32, np.float64], f"权重类型错误: {w.dtype}"
            assert b.dtype in [np.float32, np.float64], f"偏置类型错误: {b.dtype}"
        print("  ✅ 所有权重为浮点数类型")
        
        # 验证2: 数值范围
        print("\n[验证2] 检查数值范围...")
        for i, (w, b) in enumerate(zip(weights, biases)):
            w_min, w_max = w.min(), w.max()
            b_min, b_max = b.min(), b.max()
            print(f"  Layer {i+1}: W ∈ [{w_min:.4f}, {w_max:.4f}], b ∈ [{b_min:.4f}, {b_max:.4f}]")
            
            assert not np.isnan(w).any(), f"第{i+1}层权重包含NaN"
            assert not np.isnan(b).any(), f"第{i+1}层偏置包含NaN"
            assert not np.isinf(w).any(), f"第{i+1}层权重包含Inf"
            assert not np.isinf(b).any(), f"第{i+1}层偏置包含Inf"
        print("  ✅ 数值范围正常，无NaN/Inf")
        
        # 验证3: 前向传播测试
        print("\n[验证3] 前向传播测试...")
        test_input = np.random.randn(38).astype(np.float32)
        
        x = test_input
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = np.dot(x, w) + b
            
            if i < 3:  # ReLU
                x = np.maximum(0, x)
            else:  # Tanh
                x = np.tanh(x)
        
        print(f"  测试输入: {test_input[:5]}... (前5维)")
        print(f"  网络输出: {x}")
        print(f"  输出范围: [{x.min():.6f}, {x.max():.6f}]")
        assert x.shape == (2,), f"输出维度错误: {x.shape}"
        assert np.all(np.abs(x) <= 1.0), f"Tanh输出超出[-1,1]范围"
        print("  ✅ 前向传播正常")
        
        print("\n" + "="*70)
        print("✅ 权重提取验证通过！")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        print("用法: python tf_adapter.py <checkpoint目录>")
        print("示例: python tf_adapter.py ../Results/20251031_120540.356112_lidar_TD3/20251031T120540.688820_TD3_")
        sys.exit(1)
    
    verify_weight_extraction(checkpoint_dir)