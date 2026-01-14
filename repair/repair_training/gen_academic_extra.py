#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
academic_formula_pipeline.py

专门针对学术论文数学公式的数据管道
"""

import json
import random
import re
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Set
import itertools

# ============================================================================
# 学术公式相关配置
# ============================================================================

# 数学符号和运算符
MATH_SYMBOLS = {
    'greek_lower': [
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 
        'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω'
    ],
    'greek_upper': [
        'Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Υ', 'Φ', 'Ψ', 'Ω'
    ],
    'calligraphic': [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ],
    'blackboard': [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ],
    'fraktur': [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ],
    'operators': [
        '∇', '∂', '∆', '∏', '∑', '∫', '∮', '∬', '∭', '⋀', '⋁', '⋂', '⋃',
        '⨂', '⨁', '⨀', '⊗', '⊕', '⊙', '∘', '∙', '×', '÷', '±', '∓', '∧', '∨'
    ],
    'relations': [
        '=', '≡', '≠', '≈', '∼', '≅', '≃', '≜', ':=', '≐', '∝', '≤', '≥',
        '≪', '≫', '≺', '≻', '∈', '∉', '⊂', '⊃', '⊆', '⊇', '∩', '∪',
        '→', '←', '↔', '⇒', '⇐', '⇔', '↦', '⟶', '⟵', '⟹', '⟸'
    ],
    'decorations': [
        '^', '_', '̃', '̄', '̅', '⃗', '⃑', '⃡', '⃖', '⃗', '̂', '̌', '̇', '̈'
    ]
}

# 学术论文中常见的OCR错误映射
OCR_ERRORS = {
    # 括号和分隔符
    '{': ['(', '[', '｛', '〈', '‹'],
    '}': [')', ']', '｝', '〉', '›'],
    '(': ['{', '[', '〈', '‹'],
    ')': ['}', ']', '〉', '›'],
    '[': ['{', '(', '【', '〖'],
    ']': ['}', ')', '】', '〗'],
    
    # 希腊字母错误
    'π': ['n', 'TT', 'п', 'Π'],
    'θ': ['O', '0', 'Θ'],
    'φ': ['q', 'ϕ', 'Φ'],
    'ψ': ['Ψ', 'w'],
    'ω': ['w', 'W'],
    'α': ['a', 'α'],
    'β': ['B', 'β'],
    'γ': ['y', 'γ'],
    'δ': ['d', 'δ'],
    'ε': ['e', 'ε'],
    'λ': ['λ', 'A'],
    'μ': ['u', 'μ'],
    'σ': ['o', 'σ'],
    'τ': ['t', 'τ'],
    'ρ': ['p', 'ρ'],
    'ξ': ['ξ', 'Ξ'],
    'η': ['n', 'η'],
    
    # 数学符号错误
    '∑': ['E', 'Σ'],
    '∫': ['f', '∫'],
    '∂': ['d', '∂'],
    '∇': ['∆', '∇'],
    '∞': ['oo', '∞'],
    '±': ['+', '±'],
    '∓': ['-', '∓'],
    '×': ['x', '×'],
    '·': ['.', '·'],
    '⊗': ['@', '⊗'],
    '⊕': ['+', '⊕'],
    '⊙': ['o', '⊙'],
    
    # 关系符号错误
    '→': ['->', '→'],
    '⇒': ['=>', '⇒'],
    '↦': ['|->', '↦'],
    '∈': ['in', '∈'],
    '⊂': ['C', '⊂'],
    '⊆': ['c=', '⊆'],
    '≥': ['>=', '≥'],
    '≤': ['<=', '≤'],
    '≠': ['!=', '≠'],
    '≈': ['~~', '≈'],
    '∼': ['~', '∼'],
    '≅': ['~=', '≅'],
    
    # 装饰符号错误
    '̂': ['^', '̂'],
    '̄': ['-', '̄'],
    '̃': ['~', '̃'],
    '̇': ['.', '̇'],
    '̈': ['"', '̈'],
    
    # LaTeX命令错误
    '\\': ['/', '\\'],
    '^': ['^', '↑'],
    '_': ['_', '↓'],
    '&': ['&', '∧'],
    '$': ['$', 'S'],
    '%': ['%', '‰'],
    
    # 空格和不可见字符
    ' ': [' ', ' ', ' ', ' ', ' ', '　', '\t'],  # 不同宽度的空格
    '\n': ['\n', '\r\n', '\r'],
    
    # 常见混淆字符
    '0': ['O', 'o', 'θ', 'Ø'],
    '1': ['l', 'I', '|'],
    '2': ['Z', 'z'],
    '5': ['S', 's'],
    '6': ['b', 'G'],
    '8': ['B', '∞'],
    '9': ['g', 'q'],
    'l': ['1', 'I', '|'],
    'I': ['1', 'l', '|'],
    'O': ['0', 'o', 'θ'],
    'o': ['0', 'O', '°'],
    'S': ['5', 's'],
    's': ['5', 'S'],
    'Z': ['2', 'z'],
    'z': ['2', 'Z'],
}

# 学术公式模板
ACADEMIC_FORMULA_TEMPLATES = [
    # 概率与统计
    (r"P({A} \mid {B}) = \frac{{P({B} \mid {A}) P({A})}}{{P({B})}}", ["A", "B"]),
    (r"\mathbb{{E}}[{X}] = \int_\Omega {X}(\omega) \, dP(\omega)", ["X"]),
    (r"\mathrm{{Var}}({X}) = \mathbb{{E}}[({X} - \mu)^2]", ["X"]),
    (r"f_{{X}}(x) = \frac{{d}}{{dx}} F_{{X}}(x)", ["X"]),
    (r"\mathcal{{N}}(\mu, \sigma^2) = \frac{{1}}{{\sqrt{{2\pi\sigma^2}}}} e^{{-\frac{{(x-\mu)^2}}{{2\sigma^2}}}}", []),
    (r"\rho_{{XY}} = \frac{{\mathrm{{Cov}}({X},{Y})}}{{\sigma_X \sigma_Y}}", ["X", "Y"]),
    
    # 线性代数
    (r"\mathbf{{A}} \mathbf{{x}} = \lambda \mathbf{{x}}", ["A", "x"]),
    (r"\det(\mathbf{{A}} - \lambda \mathbf{{I}}) = 0", ["A"]),
    (r"\mathbf{{A}} = \mathbf{{U}} \mathbf{{\Sigma}} \mathbf{{V}}^\top", ["A"]),
    (r"\|\mathbf{{x}}\|_p = \left( \sum_{{i=1}}^n |x_i|^p \right)^{{1/p}}", ["x"]),
    (r"\langle \mathbf{{x}}, \mathbf{{y}} \rangle = \sum_{{i=1}}^n x_i \bar{{y}}_i", ["x", "y"]),
    (r"\mathbf{{A}}^\dagger = (\mathbf{{A}}^\top \mathbf{{A}})^{{-1}} \mathbf{{A}}^\top", ["A"]),
    
    # 微积分
    (r"\frac{{d}}{{d{x}}} {f}({x}) = \lim_{{h \to 0}} \frac{{{f}({x}+h) - {f}({x})}}{{h}}", ["x", "f"]),
    (r"\int_a^b {f}({x}) \, d{x} = F(b) - F(a)", ["x", "f"]),
    (r"\nabla {f} = \left( \frac{{\partial {f}}}{{\partial {x}}}, \frac{{\partial {f}}}{{\partial {y}}}, \frac{{\partial {f}}}{{\partial {z}}} \right)", ["f", "x", "y", "z"]),
    (r"\oint_C \mathbf{{F}} \cdot d\mathbf{{r}} = \iint_S (\nabla \times \mathbf{{F}}) \cdot d\mathbf{{S}}", ["F"]),
    (r"\frac{{\partial^2 {u}}}{{\partial {t}^2}} = c^2 \nabla^2 {u}", ["u", "t"]),
    (r"\mathcal{{L}}\{{f}(t)\} = \int_0^\infty e^{{-s t}} {f}(t) \, dt", ["f"]),
    
    # 优化理论
    (r"\min_{{x \in \mathbb{{R}}^n}} {f}({x}) \quad \text{{s.t.}} \quad {g}_i({x}) \leq 0, \; i=1,\dots,m", ["x", "f", "g"]),
    (r"\nabla {f}({x}^*) + \sum_{{i=1}}^m \lambda_i \nabla {g}_i({x}^*) = 0", ["x", "f", "g"]),
    (r"{x}^{{k+1}} = {x}^k - \alpha \nabla {f}({x}^k)", ["x", "f"]),
    
    # 物理公式
    (r"F = m a", []),
    (r"E = m c^2", []),
    (r"i\hbar \frac{{\partial}}{{\partial t}} \Psi = \hat{{H}} \Psi", []),
    (r"\nabla \cdot \mathbf{{E}} = \frac{{\rho}}{{\epsilon_0}}", []),
    (r"\nabla \times \mathbf{{B}} = \mu_0 \mathbf{{J}} + \mu_0 \epsilon_0 \frac{{\partial \mathbf{{E}}}}{{\partial t}}", []),
    
    # 机器学习/深度学习
    (r"\theta^* = \arg\min_\theta \mathcal{{L}}(\theta; \mathcal{{D}})", []),
    (r"\frac{{\partial \mathcal{{L}}}}{{\partial W^{{[l]}}}} = \delta^{{[l]}} (a^{{[l-1]}})^\top", []),
    (r"\mathrm{{softmax}}(z_i) = \frac{{e^{{z_i}}}}{{\sum_j e^{{z_j}}}}", []),
    (r"\mathrm{{Attention}}(Q,K,V) = \mathrm{{softmax}}\left(\frac{{QK^\top}}{{\sqrt{{d_k}}}}\right)V", []),
    
    # 量子计算
    (r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", []),
    (r"\hat{U}|\psi\rangle = e^{{-i\hat{H}t/\hbar}}|\psi\rangle", []),
    (r"\mathrm{Tr}(\rho\hat{O}) = \langle\hat{O}\rangle", []),
    
    # 控制理论
    (r"\dot{x} = Ax + Bu", []),
    (r"y = Cx + Du", []),
    (r"u(t) = -Kx(t)", []),
    
    # 信息论
    (r"H(X) = -\sum_{i=1}^n p(x_i) \log_2 p(x_i)", []),
    (r"I(X;Y) = H(X) - H(X|Y)", []),
    (r"C = \max_{p(x)} I(X;Y)", []),
]

# 变量集合
VARIABLES = {
    'scalars': ['x', 'y', 'z', 't', 's', 'u', 'v', 'w', 'r', 'θ', 'φ', 'ψ', 'λ', 'μ', 'σ', 'τ'],
    'vectors': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q'],
    'matrices': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
    'sets': ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ω', 'Λ', 'Σ', 'Π', 'Φ', 'Ψ'],
    'functions': ['f', 'g', 'h', 'F', 'G', 'H', 'φ', 'ψ', 'χ', 'ξ', 'η', 'ζ'],
}

# ============================================================================
# 专用数据生成函数
# ============================================================================

def introduce_ocr_errors(text: str, error_rate: float = 0.05) -> str:
    """引入OCR错误"""
    if not text or error_rate <= 0:
        return text
    
    result = []
    for char in text:
        if random.random() < error_rate and char in OCR_ERRORS:
            # 引入错误
            possible_errors = OCR_ERRORS[char]
            if possible_errors:
                result.append(random.choice(possible_errors))
            else:
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)

def introduce_latex_errors(text: str, error_rate: float = 0.03) -> str:
    """引入LaTeX特定的错误"""
    errors = []
    
    # LaTeX命令错误
    if random.random() < error_rate:
        errors.append(lambda s: s.replace(r'\begin{equation}', r'\beginequation'))
    if random.random() < error_rate:
        errors.append(lambda s: s.replace(r'\end{equation}', r'\endequation'))
    if random.random() < error_rate:
        errors.append(lambda s: s.replace(r'\mathcal', r'\mathical'))
    if random.random() < error_rate:
        errors.append(lambda s: s.replace(r'\mathbb', r'\mathb'))
    if random.random() < error_rate:
        errors.append(lambda s: s.replace(r'\{', '('))
    if random.random() < error_rate:
        errors.append(lambda s: s.replace(r'\}', ')'))
    
    # 多余的空白
    if random.random() < error_rate:
        errors.append(lambda s: re.sub(r'\s+', '  ', s))  # 双空格
    
    # 缺失的括号
    if random.random() < error_rate:
        errors.append(lambda s: s.replace('(', '').replace(')', ''))
    
    result = text
    for error_func in errors:
        result = error_func(result)
    
    return result

def introduce_typing_errors(text: str, error_rate: float = 0.02) -> str:
    """引入打字错误"""
    if not text or error_rate <= 0:
        return text
    
    result = list(text)
    n = len(result)
    
    # 字符替换错误
    for i in range(n):
        if random.random() < error_rate:
            char = result[i]
            # 附近键盘位置替换
            if char == 'a': result[i] = random.choice(['s', 'q', 'w', 'z'])
            elif char == 'b': result[i] = random.choice(['v', 'g', 'h', 'n'])
            elif char == 'c': result[i] = random.choice(['x', 'd', 'f', 'v'])
            elif char == 'd': result[i] = random.choice(['s', 'e', 'r', 'f', 'c'])
            elif char == 'e': result[i] = random.choice(['w', 'r', 's', 'd'])
            elif char == 'f': result[i] = random.choice(['d', 'r', 't', 'g', 'c', 'v'])
            elif char == 'g': result[i] = random.choice(['f', 't', 'y', 'h', 'v', 'b'])
            elif char == 'h': result[i] = random.choice(['g', 'y', 'u', 'j', 'b', 'n'])
            elif char == 'i': result[i] = random.choice(['u', 'o', 'j', 'k'])
            elif char == 'j': result[i] = random.choice(['h', 'u', 'i', 'k', 'n', 'm'])
            elif char == 'k': result[i] = random.choice(['j', 'i', 'o', 'l', 'm'])
            elif char == 'l': result[i] = random.choice(['k', 'o', 'p', ';'])
            elif char == 'm': result[i] = random.choice(['n', 'j', 'k', 'l'])
            elif char == 'n': result[i] = random.choice(['b', 'h', 'j', 'k', 'm'])
            elif char == 'o': result[i] = random.choice(['i', 'p', 'k', 'l'])
            elif char == 'p': result[i] = random.choice(['o', 'l', ';'])
            elif char == 'q': result[i] = random.choice(['w', 'a'])
            elif char == 'r': result[i] = random.choice(['e', 't', 'd', 'f'])
            elif char == 's': result[i] = random.choice(['a', 'w', 'e', 'd', 'x', 'z'])
            elif char == 't': result[i] = random.choice(['r', 'y', 'f', 'g'])
            elif char == 'u': result[i] = random.choice(['y', 'i', 'h', 'j'])
            elif char == 'v': result[i] = random.choice(['c', 'f', 'g', 'b'])
            elif char == 'w': result[i] = random.choice(['q', 'e', 'a', 's'])
            elif char == 'x': result[i] = random.choice(['z', 's', 'd', 'c'])
            elif char == 'y': result[i] = random.choice(['t', 'u', 'g', 'h'])
            elif char == 'z': result[i] = random.choice(['a', 's', 'x'])
    
    # 字符缺失错误
    result_text = ''.join(result)
    if random.random() < error_rate and len(result_text) > 1:
        idx = random.randint(0, len(result_text) - 1)
        result_text = result_text[:idx] + result_text[idx+1:]
    
    # 字符重复错误
    if random.random() < error_rate and len(result_text) > 1:
        idx = random.randint(0, len(result_text) - 1)
        result_text = result_text[:idx] + result_text[idx] + result_text[idx:]
    
    # 交换相邻字符
    if random.random() < error_rate and len(result_text) > 2:
        idx = random.randint(0, len(result_text) - 2)
        chars = list(result_text)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        result_text = ''.join(chars)
    
    return result_text

def wrap_in_latex_document(formula: str) -> str:
    """将公式包装在完整的LaTeX文档中"""
    # 随机选择LaTeX文档结构
    doc_templates = [
        # 学术论文风格
        r"""\documentclass{{article}}
\usepackage{{amsmath, amssymb, amsfonts}}
\usepackage{{bm}}
\begin{{document}}

{formula}

\end{{document}}""",
        
        # 会议论文风格
        r"""\documentclass{{llncs}}
\usepackage{{amsmath, amssymb}}
\begin{{document}}

{formula}

\end{{document}}""",
        
        # 简单风格
        r"""\documentclass{{article}}
\usepackage{{amsmath}}
\begin{{document}}

{formula}

\end{{document}}""",
        
        # 包含图形和表格
        r"""\documentclass{{article}}
\usepackage{{amsmath, graphicx}}
\usepackage{{booktabs}}
\begin{{document}}

{formula}

\end{{document}}""",
    ]
    
    template = random.choice(doc_templates)
    return template.format(formula=formula)

def generate_academic_formula_data(count: int) -> List[Dict]:
    """生成学术论文公式数据"""
    print(f"生成 {count} 条学术公式数据...")
    
    formulas = []
    
    for i in range(count):
        # 选择模板
        template, params = random.choice(ACADEMIC_FORMULA_TEMPLATES)
        
        # 为参数选择合适的变量
        replacements = {}
        for param in params:
            # 根据参数类型选择合适的变量
            if param.lower() == param and len(param) == 1:  # 小写单字母，可能是标量
                replacements[param] = random.choice(VARIABLES['scalars'])
            elif param.upper() == param and len(param) == 1:  # 大写单字母，可能是矩阵
                replacements[param] = random.choice(VARIABLES['matrices'])
            else:  # 其他情况
                replacements[param] = random.choice(VARIABLES['functions'])
        
        # 应用替换
        clean_formula = template
        for param, var in replacements.items():
            clean_formula = clean_formula.replace("{" + param + "}", var)
        
        # 随机选择是否添加LaTeX环境
        latex_envs = [
            ("equation", r"\begin{equation}" + clean_formula + r"\end{equation}"),
            ("equation*", r"\begin{equation*}" + clean_formula + r"\end{equation*}"),
            ("align", r"\begin{align}" + clean_formula + r"\end{align}"),
            ("align*", r"\begin{align*}" + clean_formula + r"\end{align*}"),
            ("gather", r"\begin{gather}" + clean_formula + r"\end{gather}"),
            ("multline", r"\begin{multline}" + clean_formula + r"\end{multline}"),
        ]
        
        if random.random() < 0.7:  # 70%的概率添加环境
            env_name, clean_latex = random.choice(latex_envs)
        else:
            clean_latex = clean_formula
        
        # 40%的概率包装在完整文档中
        if random.random() < 0.4:
            clean_latex = wrap_in_latex_document(clean_latex)
        
        # 引入各种错误
        noisy_latex = clean_latex
        
        # 应用OCR错误（15%概率）
        if random.random() < 0.15:
            noisy_latex = introduce_ocr_errors(noisy_latex, error_rate=0.05)
        
        # 应用LaTeX错误（10%概率）
        if random.random() < 0.10:
            noisy_latex = introduce_latex_errors(noisy_latex, error_rate=0.03)
        
        # 应用打字错误（8%概率）
        if random.random() < 0.08:
            noisy_latex = introduce_typing_errors(noisy_latex, error_rate=0.02)
        
        # 额外的随机错误
        if random.random() < 0.05:
            # 添加随机空格
            if ' ' in noisy_latex:
                noisy_latex = noisy_latex.replace(' ', '  ', random.randint(1, 3))
        
        if random.random() < 0.03:
            # 移除必要的括号
            noisy_latex = noisy_latex.replace('(', '').replace(')', '').replace('{', '').replace('}', '')
        
        formulas.append({
            "input": noisy_latex,
            "output": clean_latex,
            "type": "formula",
            "template": template,
            "error_type": "mixed" if noisy_latex != clean_latex else "none"
        })
        
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{count} 条")
    
    return formulas

def generate_specific_errors_from_examples() -> List[Dict]:
    """从您提供的示例中生成特定错误模式"""
    print("从示例中生成特定错误数据...")
    
    examples = [
        # 示例1: 箭头和括号错误
        {
            "clean": r"\begin{equation} \mathcal{D}^\pi(\mathbf{s}, \Omega) \longrightarrow \mathbb{E}[\mathcal{B} \mid S, \Omega, \mathcal{T}] \end{equation}",
            "errors": [
                r"\begin{equation} \mathcal{D}^\pi(\mathbf{s}, \Omega) \longrightarrow \mathbb{E}[\mathcal{B} \mid S, \Omega, \mathcal{T}] \end{equation}",  # 正确
                r"\begin{equation} \mathcal{D}^\pi(\mathbf{s}, \Omega) -> \mathbb{E}[\mathcal{B} | S, \Omega, \mathcal{T}] \end{equation}",  # 箭头和竖线错误
                r"\begin{equation} \mathcal{D}^\pi(\mathbf{s}, \Omega) \rightarrow \mathbb{E}[\mathcal{B} \mid S, \Omega, \mathcal{T}] \end{equation}",  # 箭头变体
                r"\begin{equation} \mathcal{D}^{\pi}(\mathbf{s},\,\Omega)\longrightarrow\,\mathbb{E}\big[\mathcal{B}\big\vert S,\,\Omega,\,\mathcal{T}\big] \end{equation}",  # 间距和大括号
            ]
        },
        
        # 示例2: 张量符号错误
        {
            "clean": r"\begin{equation} \bigotimes^\pi(S_t, \Omega_t) \hookrightarrow (N_t, \Omega_t) \sim \sum_Q \hat{O} \end{equation}",
            "errors": [
                r"\begin{equation} \bigotimes^\pi(S_t, \Omega_t) \hookrightarrow (N_t, \Omega_t) \sim \sum_Q \hat{O} \end{equation}",
                r"\begin{equation} \bigotimes^{\pi}\left(\S_{t}\,,\,\Omega_{t}\right)\,\stackrel{\sim}{\hookrightarrow}\,\left(\n\n_{t}\,,\,\Omega_{t}\right)\,\stackrel{\ldots}{\sim}\,\sum_{Q}\hat{O} \end{equation}",  # 您提供的错误示例
                r"\begin{equation} \bigotimes^{\pi}(S_t, \Omega_t) \sim \hookrightarrow (N_t, \Omega_t) ... \sum_Q \hat{O} \end{equation}",
                r"\begin{equation} \otimes^\pi(S_t, \Omega_t) \rightarrow (N_t, \Omega_t) ~ \sum_Q O \end{equation}",
            ]
        },
        
        # 示例3: 符号混淆
        {
            "clean": r"\begin{equation} V^\pi(\mathbf{s}) = \mathbb{R}[N](\mathbf{s}, \mathbb{T}) \end{equation}",
            "errors": [
                r"\begin{equation} V^\pi(\mathbf{s}) = \mathbb{R}[N](\mathbf{s}, \mathbb{T}) \end{equation}",
                r"\begin{equation} V^{\pi}(\mathbf{s})\ --\mathbb{R}[N]\mathbf{s},\,\mathbb{T}] \end{equation}",  # 您提供的错误示例
                r"\begin{equation} V^\pi(s) = R[N](s, T) \end{equation}",
                r"\begin{equation} V^{\pi}(\mathbf{s}) = \mathbb{R}[N](\mathbf{s}, \mathbb{T} \end{equation}",  # 缺失括号
            ]
        },
        
        # 示例4: 希腊字母错误
        {
            "clean": r"\begin{equation} \tau(6) \end{equation}",
            "errors": [
                r"\begin{equation} \tau(6) \end{equation}",
                r"\begin{equation} \mathbf{\tau}(6) \end{equation}",  # 粗体错误
                r"\begin{equation} t(6) \end{equation}",  # 字母替换
                r"\begin{equation} \tau 6 \end{equation}",  # 缺失括号
            ]
        },
        
        # 示例5: Psi符号错误
        {
            "clean": r"\begin{equation} \Psi(5) \end{equation}",
            "errors": [
                r"\begin{equation} \Psi(5) \end{equation}",
                r"\begin{equation} \mathbf{\Psi}(5) \end{equation}",  # 您提供的错误示例
                r"\begin{equation} \psi(5) \end{equation}",  # 大小写错误
                r"\begin{equation} Psi(5) \end{equation}",  # 命令缺失
            ]
        },
    ]
    
    data = []
    for example in examples:
        clean = example["clean"]
        for error_version in example["errors"]:
            data.append({
                "input": error_version,
                "output": clean,
                "type": "formula",
                "template": "example_based",
                "error_type": "specific_example"
            })
    
    return data

def generate_academic_noise_data(count: int) -> List[Dict]:
    """生成学术文本噪声数据"""
    print(f"生成 {count} 条学术噪声数据...")
    
    noise_data = []
    
    # 学术文本片段
    academic_snippets = [
        # 引用和标注
        "See Fig. 1 for a detailed illustration of the proposed framework.",
        "As shown in Table 2, the experimental results demonstrate significant improvements.",
        "Previous work by Smith et al. [1] established the theoretical foundations.",
        "This approach builds upon the methodology introduced in [2,3].",
        "We refer the reader to Section 4 for implementation details.",
        "The proof follows directly from Lemma 3.2 and Theorem 4.1.",
        
        # 实验设置
        "All experiments were conducted on a Linux server with 8 NVIDIA V100 GPUs.",
        "We implemented our model using PyTorch 1.9.0 and Python 3.8.",
        "The training process took approximately 72 hours to converge.",
        "We used the Adam optimizer with default parameters and a learning rate of 0.001.",
        "The dataset was randomly split into training (80%), validation (10%), and test (10%) sets.",
        "We employed 5-fold cross-validation to ensure statistical significance.",
        
        # 数学描述
        "Without loss of generality, we assume that the underlying distribution is Gaussian.",
        "The objective function is non-convex but differentiable almost everywhere.",
        "By applying Jensen's inequality, we obtain the following upper bound.",
        "The convergence rate is O(1/√T), where T denotes the number of iterations.",
        "This can be formulated as a constrained optimization problem.",
        "The solution satisfies the Karush-Kuhn-Tucker conditions.",
        
        # 结果讨论
        "Our method achieves state-of-the-art performance on all benchmark datasets.",
        "The improvement over the baseline is statistically significant (p < 0.01).",
        "Ablation studies confirm the contribution of each component.",
        "The results are robust to variations in hyperparameter settings.",
        "We observed diminishing returns when increasing model capacity beyond this point.",
        "The error bars represent one standard deviation across five independent runs.",
        
        # 限制和未来工作
        "One limitation of our approach is its sensitivity to initialization.",
        "Future work could extend this framework to multi-modal settings.",
        "The computational complexity scales quadratically with input size.",
        "We leave the theoretical analysis of convergence properties for future work.",
        "The method assumes independence between features, which may not hold in practice.",
        "Extending this to non-Euclidean domains remains an open challenge.",
    ]
    
    # LaTeX特定的噪声
    latex_noise = [
        r"\usepackage{amsmath,amssymb}",
        r"\bibliographystyle{ieeetr}",
        r"\title{On the Convergence of Stochastic Gradient Methods}",
        r"\author{Jane Doe$^1$, John Smith$^2$}",
        r"\affiliation{$^1$University A, $^2$Institute B}",
        r"\date{\today}",
        r"\maketitle",
        r"\abstract{This paper presents a novel approach to...}",
        r"\keywords{machine learning, optimization, deep learning}",
        r"\section{Introduction}",
        r"\subsection{Related Work}",
        r"\label{fig:architecture}",
        r"\ref{tab:results}",
        r"\cite{smith2021deep}",
        r"\footnote{Code available at https://github.com/}",
        r"\caption{Comparison of different methods.}",
        r"\label{eq:main}",
        r"\item First item in the list",
        r"\begin{itemize}",
        r"\end{itemize}",
        r"\begin{enumerate}",
        r"\end{enumerate}",
        r"\begin{table}[htbp]",
        r"\end{table}",
        r"\begin{figure}[htbp]",
        r"\end{figure}",
        r"\centering",
        r"\includegraphics[width=0.8\textwidth]{figure.pdf}",
        r"\hline",
        r"\toprule",
        r"\midrule",
        r"\bottomrule",
    ]
    
    for i in range(count):
        # 随机组合噪声类型
        if random.random() < 0.6:
            # 纯文本噪声
            num_snippets = random.randint(1, 3)
            selected = random.sample(academic_snippets, num_snippets)
            noise_text = " ".join(selected)
            
            # 可能添加LaTeX包装
            if random.random() < 0.3:
                noise_text = rf"\begin{{minipage}}{{\textwidth}} {noise_text} \end{{minipage}}"
        
        else:
            # LaTeX命令噪声
            noise_text = random.choice(latex_noise)
            
            # 可能包装在环境中
            if random.random() < 0.4:
                env = random.choice(["center", "flushleft", "flushright", "quote"])
                noise_text = rf"\begin{{{env}}} {noise_text} \end{{{env}}}"
        
        # 引入错误
        if random.random() < 0.2:
            noise_text = introduce_typing_errors(noise_text, error_rate=0.03)
        
        noise_data.append({
            "input": noise_text,
            "output": "noise",
            "type": "academic_noise",
            "subtype": "text" if "\\begin{" not in noise_text else "latex_env"
        })
        
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{count} 条")
    
    return noise_data

# ============================================================================
# 数据整合和数据集创建（与原脚本相同）
# ============================================================================

def load_all_existing_data(source_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """加载所有已有数据"""
    print(f"\n加载已有数据从: {source_dir}")
    
    all_formulas = []
    all_short_noise = []
    all_long_noise = []
    
    if not source_dir.exists():
        print(f"警告: 目录不存在 {source_dir}")
        return all_formulas, all_short_noise, all_long_noise
    
    # 查找所有JSONL文件
    jsonl_files = list(source_dir.glob("*.jsonl"))
    
    for filepath in jsonl_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_data = []
                for line in f:
                    if line.strip():
                        try:
                            file_data.append(json.loads(line.strip()))
                        except:
                            continue
                
                # 分类
                for item in file_data:
                    output = item.get("output", "").strip().lower()
                    input_text = item.get("input", "").strip()
                    
                    if output == "noise":
                        if len(input_text) < 100:  # 短噪声
                            all_short_noise.append(item)
                        else:  # 长噪声
                            all_long_noise.append(item)
                    else:
                        all_formulas.append(item)
                
                print(f"  ✓ {filepath.name}: {len(file_data)} 条")
                
        except Exception as e:
            print(f"  ✗ {filepath.name}: 加载失败 - {e}")
    
    return all_formulas, all_short_noise, all_long_noise
def create_academic_dataset(
    existing_formulas: List[Dict],
    existing_short: List[Dict],
    existing_long: List[Dict],
    new_formulas: List[Dict],
    new_short: List[Dict],
    new_long: List[Dict],
    output_dir: Path
):
    """根据你的要求，只输出三类分开的文件：
       formulas.jsonl (8040)
       short_noise.jsonl (6650)
       long_noise.jsonl (18)
    """

    print("\n" + "="*80)
    print("创建学术专用数据集（按你指定数量分开保存）")
    print("="*80)

    # -----------------------------------------
    # 1. 拼接三类数据
    # -----------------------------------------
    final_formulas = existing_formulas + new_formulas
    final_short_noise = existing_short + new_short
    final_long_noise = existing_long + new_long

    print(f"公式数据（目标8040）: {len(final_formulas)} 条")
    print(f"短噪声（目标6650）: {len(final_short_noise)} 条")
    print(f"长噪声（目标18）: {len(final_long_noise)} 条")

    # -----------------------------------------
    # 2. 保存到目录
    # -----------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(file_name, data):
        path = output_dir / file_name
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  ✓ 已保存 {file_name}: {len(data)} 条")

    # 只保存你需要的三类
    save_jsonl("formulas.jsonl", final_formulas)
    save_jsonl("short_noise.jsonl", final_short_noise)
    save_jsonl("long_noise.jsonl", final_long_noise)

    print("\n数据已按要求生成完毕。")
    print(f"保存目录: {output_dir}")

    return {
        "formulas": final_formulas,
        "short_noise": final_short_noise,
        "long_noise": final_long_noise
    }

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="学术论文公式数据管道：专门针对OCR和LaTeX错误"
    )
    
    parser.add_argument("--generate_new", action="store_true",
                       help="是否生成新数据")
    parser.add_argument("--new_formulas", type=int, default=2000,
                       help="新生成公式数量")
    parser.add_argument("--new_academic_noise", type=int, default=500,
                       help="新生成学术噪声数量")
    
    parser.add_argument("--include_examples", action="store_true",
                       help="包含示例中的特定错误模式")
    
    parser.add_argument("--existing_dir", type=str, 
                       default="data/train_dataset_cleaned",
                       help="已有数据目录")
    
    parser.add_argument("--output_dir", type=str,
                       default="data/academic_formula_dataset",
                       help="输出目录")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    print("=" * 80)
    print("学术论文公式数据管道")
    print("=" * 80)
    
    # 生成新数据
    new_formulas = []
    new_academic_noise = []
    
    if args.generate_new:
        print("\n步骤1: 生成新数据")
        print("-" * 40)
        
        try:
            new_formulas = generate_academic_formula_data(args.new_formulas)
            new_academic_noise = generate_academic_noise_data(args.new_academic_noise)
            
            print(f"✓ 生成完成:")
            print(f"  学术公式: {len(new_formulas)} 条")
            print(f"  学术噪声: {len(new_academic_noise)} 条")
        except Exception as e:
            print(f"✗ 生成数据时出错: {e}")
    
    # 包含示例数据
    if args.include_examples:
        print("\n步骤1.5: 包含示例错误模式")
        print("-" * 40)
        
        example_data = generate_specific_errors_from_examples()
        new_formulas.extend(example_data)
        print(f"✓ 示例数据: {len(example_data)} 条")
    
    # 加载已有数据
    print("\n步骤2: 加载已有数据")
    print("-" * 40)
    
    try:
        existing_formulas, existing_short, existing_long = load_all_existing_data(
            Path(args.existing_dir)
        )
        
        print(f"✓ 加载完成:")
        print(f"  现有公式: {len(existing_formulas)} 条")
        print(f"  现有短噪声: {len(existing_short)} 条")
        print(f"  现有长噪声: {len(existing_long)} 条")
    except Exception as e:
        print(f"✗ 加载数据时出错: {e}")
        existing_formulas, existing_short, existing_long = [], [], []
    
    # 创建数据集
    print("\n步骤3: 创建学术数据集")
    print("-" * 40)
    
    try:
        all_data = create_academic_dataset(
            existing_formulas=existing_formulas,
            existing_short=existing_short,
            existing_long=existing_long,
            new_formulas=new_formulas,
            new_short=new_academic_noise,  # 使用学术噪声
            new_long=[],  # 长短噪声合并处理
            output_dir=Path(args.output_dir)
        )
        
        # 总结
        print("\n" + "=" * 80)
        print("管道完成!")
        print("=" * 80)
        
        print(f"\n总结:")
        print(f"  总数据量: {len(all_data)} 条")
        print(f"  公式数据: {len(existing_formulas) + len(new_formulas)} 条")
        print(f"  噪声数据: {len(existing_short) + len(existing_long) + len(new_academic_noise)} 条")
        
        print(f"\n生成的文件 ({args.output_dir}):")
        
        output_path = Path(args.output_dir)
        for file in output_path.glob("*.jsonl"):
            with open(file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"  ✓ {file.name}: {line_count} 行")

    except Exception as e:
        print(f"\n✗ 创建数据集时出错: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 使用说明
# ============================================================================

if __name__ == "__main__":
    print("""
学术论文公式数据管道 - 专门针对OCR和LaTeX错误

使用示例:

1. 只整合已有数据:
   python academic_formula_pipeline.py \\
     --existing_dir data/train_dataset_cleaned \\
     --output_dir data/academic_dataset

2. 生成新数据并整合:
   python academic_formula_pipeline.py \\
     --generate_new \\
     --new_formulas 2500 \\
     --new_academic_noise 500 \\
     --existing_dir data/train_dataset_cleaned \\
     --output_dir data/academic_dataset_with_new

3. 包含您的示例错误模式:
   python academic_formula_pipeline.py \\
     --generate_new \\
     --include_examples \\
     --new_formulas 1000 \\
     --new_academic_noise 300 \\
     --output_dir data/academic_dataset_examples

4. 快速测试:
   python academic_formula_pipeline.py \\
     --generate_new \\
     --new_formulas 10 \\
     --new_academic_noise 5 \\
     --output_dir data/test_output
    """)
    print("\n" + "="*80)
    
    main()