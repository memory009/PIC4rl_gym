#!/usr/bin/env python3
"""
POLAR Taylor Model 可达性验证核心模块
严格遵循论文: "Reachability Verification Based Reliability Assessment"
"""

import numpy as np
import sympy as sym
from functools import reduce
import operator as op
import math


def ncr(n, r):
    """组合数 C(n, r)"""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


class TaylorModel:
    """
    Taylor模型：多项式 + 误差区间
    
    数学定义 (论文 Section III-B):
        TM(z) = p(z) + I, z ∈ [-1, 1]^n
    
    属性:
        poly: sympy.Poly - 多项式部分 p(z)
        remainder: [I⁻, I⁺] - 误差区间
    """
    def __init__(self, poly, remainder):
        self.poly = poly
        self.remainder = remainder


class TaylorArithmetic:
    """Taylor算术运算 (论文 Equation 6)"""
    
    def weighted_sumforall(self, taylor_models, weights, bias):
        """
        加权和：Σ(wᵢ * TMᵢ) + b
        
        论文公式:
            TM_out = Σᵢ wᵢ * TM_in[i] + b
            多项式: p_out = Σᵢ wᵢ * p_in[i] + b
            误差: I_out = Σᵢ |wᵢ| * |I_in[i]|
        """
        # 多项式求和
        temp_poly = 0
        for i, tm in enumerate(taylor_models):
            temp_poly += weights[i] * tm.poly
        temp_poly += bias
        
        # 误差累积 (取绝对值保证保守性)
        temp_remainder = 0
        for i, tm in enumerate(taylor_models):
            temp_remainder += abs(weights[i]) * tm.remainder[1]
        
        remainder = [-temp_remainder, temp_remainder]
        
        # 确保是 Poly 对象
        if not isinstance(temp_poly, sym.Poly):
            if hasattr(taylor_models[0].poly, 'gens') and taylor_models[0].poly.gens:
                temp_poly = sym.Poly(temp_poly, *taylor_models[0].poly.gens)
            else:
                temp_poly = sym.Poly(temp_poly)
        
        return TaylorModel(temp_poly, remainder)
    
    def constant_product(self, taylor_model, constant):
        """
        常数乘法：c * TM
        
        论文: TM_out = c * TM_in
        """
        new_poly = constant * taylor_model.poly
        new_remainder = [
            constant * taylor_model.remainder[0],
            constant * taylor_model.remainder[1]
        ]
        
        # 确保是 Poly 对象
        if not isinstance(new_poly, sym.Poly):
            if hasattr(taylor_model.poly, 'gens') and taylor_model.poly.gens:
                new_poly = sym.Poly(new_poly, *taylor_model.poly.gens)
            else:
                new_poly = sym.Poly(new_poly)
        
        return TaylorModel(new_poly, new_remainder)


class BernsteinPolynomial:
    """
    Bernstein多项式逼近激活函数
    
    论文 Equation (1):
        p_σ = Σᵢ₌₀ᵏ σ(aᵢ) * C(k,i) * Bᵢ(y)
    
    其中 Bᵢ(y) 是 Bernstein 基函数
    """
    
    def __init__(self, error_steps=4000):
        """
        Args:
            error_steps: 误差估计采样点数（论文使用4000）
        """
        self.error_steps = error_steps
        self.bern_poly = None
    
    def approximate(self, a, b, order, activation_name):
        """
        在区间 [a, b] 上用 Bernstein 多项式逼近激活函数
        
        论文公式 (1):
            p_σ = Σᵢ₌₀ᵏ σ(a + (b-a)/k * i) * C(k,i) * Bᵢ(y)
        """
        # 自适应阶数选择
        d_max = 8
        d_p = np.floor(d_max / math.log10(1 / (b - a)))
        d_p = np.abs(d_p)
        if d_p < 2:
            d_p = 2
        d = min(order, d_p)
        
        # 映射系数：将 [a, b] 映射到 [0, 1]
        coe1_1 = -a / (b - a)
        coe1_2 = 1 / (b - a)
        coe2_1 = b / (b - a)
        coe2_2 = -1 / (b - a)
        
        x = sym.Symbol('x')
        bern_poly = 0 * x
        
        # Bernstein 基函数展开
        for v in range(int(d) + 1):
            c = ncr(int(d), v)
            point = a + (b - a) / d * v
            
            # 计算激活函数值
            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(float(point))
            else:
                raise ValueError(f"不支持的激活函数: {activation_name}")
            
            # Bernstein 基
            basis = (
                ((coe1_2 * x + coe1_1) ** v) * 
                ((coe2_1 + coe2_2 * x) ** (d - v))
            )
            
            bern_poly += c * f_value * basis
        
        if bern_poly == 0:
            bern_poly = 1e-16 * x
        
        self.bern_poly = bern_poly
        return sym.Poly(bern_poly)
    
    def compute_error(self, a, b, activation_name):
        """
        计算 Bernstein 近似的误差上界
        
        论文公式 (2):
            ε = max_{i=0,...,m} |p_σ(sᵢ) - σ(sᵢ)| + (b-a)/m
        """
        epsilon = 0
        m = self.error_steps
        
        for v in range(m + 1):
            # 采样点
            point = a + (b - a) / m * (v + 0.5)
            
            # 真实激活函数值
            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(float(point))
            
            # Bernstein 多项式值
            b_value = sym.Poly(self.bern_poly).eval(point)
            
            # 更新最大误差
            temp_diff = abs(f_value - b_value)
            epsilon = max(epsilon, temp_diff)
        
        # 加上离散化误差
        return epsilon + (b - a) / m


def compute_tm_bounds(tm):
    """
    计算 Taylor 模型的上下界
    
    论文 Minkowski 加法 (Equation 7):
        P ⊕ I = {p + i | p ∈ P, i ∈ I}
    
    保守估计方法:
        - 常数项：精确累加
        - 变量项：取绝对值（因为 z ∈ [-1, 1]）
    """
    poly = tm.poly
    
    temp_upper = 0
    temp_lower = 0
    
    # 遍历多项式的每一项
    for i in range(len(poly.monoms())):
        coeff = poly.coeffs()[i]
        monom = poly.monoms()[i]
        
        if sum(monom) == 0:  # 常数项
            temp_upper += coeff
            temp_lower += coeff
        else:  # 变量项
            temp_upper += abs(coeff)
            temp_lower += -abs(coeff)
    
    # 加上误差区间
    a = temp_lower + tm.remainder[0]
    b = temp_upper + tm.remainder[1]
    
    return float(a), float(b)


def compute_poly_bounds(poly):
    """
    计算多项式的上下界（不考虑误差区间）
    """
    temp_upper = 0
    temp_lower = 0
    
    for i in range(len(poly.monoms())):
        coeff = poly.coeffs()[i]
        monom = poly.monoms()[i]
        
        if sum(monom) == 0:
            temp_upper += coeff
            temp_lower += coeff
        else:
            temp_upper += abs(coeff)
            temp_lower += -abs(coeff)
    
    return float(temp_lower), float(temp_upper)


def apply_activation(tm, bern_poly, bern_error, max_order):
    """
    通过 Bernstein 多项式传播 Taylor 模型过激活函数
    
    步骤:
        1. 多项式合成: p_out = p_bern(p_in)
        2. 截断到指定阶数
        3. 计算截断误差
        4. 累积 Bernstein 近似误差
    """
    # 1. 多项式合成
    composed = sym.polys.polytools.compose(bern_poly, tm.poly)
    
    # 2. 截断到 max_order
    poly_truncated = 0
    for i in range(len(composed.monoms())):
        monom = composed.monoms()[i]
        if sum(monom) <= max_order:
            temp = 1
            for j in range(len(monom)):
                temp *= composed.gens[j] ** monom[j]
            poly_truncated += composed.coeffs()[i] * temp
    
    poly_truncated = sym.Poly(poly_truncated)
    
    # 3. 计算截断误差
    poly_remainder = composed - poly_truncated
    _, truncation_error = compute_poly_bounds(poly_remainder)
    
    # 4. 计算总误差
    total_remainder = 0
    
    # Bernstein 多项式对输入误差的传播
    for i in range(len(bern_poly.monoms())):
        monom = bern_poly.monoms()[i]
        degree = sum(monom)
        
        if degree < 1:
            continue
        elif degree == 1:
            total_remainder += abs(bern_poly.coeffs()[i] * tm.remainder[1])
        else:
            # 高阶项：误差会被放大
            total_remainder += abs(bern_poly.coeffs()[i] * (tm.remainder[1] ** degree))
    
    # 总误差 = 传播误差 + 截断误差 + Bernstein误差
    remainder = [
        -total_remainder - truncation_error - bern_error,
        total_remainder + truncation_error + bern_error
    ]
    
    return TaylorModel(poly_truncated, remainder)