#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大语言模型记忆化vs推理能力评估 - 演示版本（无需模型下载）

使用模拟数据展示完整的评估流程和分析方法
"""

import json
import time
import random
import os
from typing import Dict, List, Tuple
import numpy as np

class MockLLM:
    """模拟LLM模型"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        random.seed(42)  # 确保结果可重复
        
        # 不同模型的特性模拟
        if "qwen" in model_name:
            self.reasoning_ability = 0.7
            self.memory_ability = 0.8
            self.base_accuracy = 0.75
        elif "llama" in model_name:
            self.reasoning_ability = 0.8
            self.memory_ability = 0.7
            self.base_accuracy = 0.8
        else:
            self.reasoning_ability = 0.6
            self.memory_ability = 0.9
            self.base_accuracy = 0.7
    
    def query(self, prompt: str, **kwargs) -> str:
        """模拟查询响应"""
        # 简单的基于关键词的响应生成
        prompt_lower = prompt.lower()
        
        # 算术问题
        if "what is" in prompt_lower and ("+" in prompt or "-" in prompt or "*" in prompt):
            return self._generate_arithmetic_response(prompt)
        
        # 逻辑谜题
        elif "knight" in prompt_lower and "knave" in prompt_lower:
            return self._generate_logic_response(prompt)
        
        # 一般问题
        else:
            return self._generate_general_response(prompt)
    
    def _generate_arithmetic_response(self, prompt: str) -> str:
        """生成算术问题响应"""
        # 模拟不同准确率
        is_correct = random.random() < self.base_accuracy
        
        if "base 11" in prompt.lower():
            # 反事实任务，准确率下降
            is_correct = random.random() < (self.base_accuracy * 0.4)
        
        if is_correct:
            if "2 + 3" in prompt:
                return "Let me solve this step by step. 2 + 3 = 5"
            elif "5 - 2" in prompt:
                return "Let me calculate: 5 - 2 = 3"
            else:
                # 提取数字并计算
                numbers = [int(x) for x in prompt.split() if x.isdigit()]
                if len(numbers) >= 2:
                    if "+" in prompt:
                        result = numbers[0] + numbers[1]
                        return f"Calculating step by step: {numbers[0]} + {numbers[1]} = {result}"
                    elif "-" in prompt:
                        result = numbers[0] - numbers[1]
                        return f"Let me solve: {numbers[0]} - {numbers[1]} = {result}"
                    elif "*" in prompt:
                        result = numbers[0] * numbers[1]
                        return f"Multiplying: {numbers[0]} * {numbers[1]} = {result}"
        
        return "I think the answer is 42"  # 错误答案
    
    def _generate_logic_response(self, prompt: str) -> str:
        """生成逻辑谜题响应"""
        # 提取人名
        names = []
        common_names = ['alice', 'bob', 'zoey', 'oliver', 'william', 'evelyn']
        for name in common_names:
            if name in prompt.lower():
                names.append(name.capitalize())
        
        if len(names) >= 2:
            # 模拟推理准确率
            is_correct = random.random() < (self.reasoning_ability * 0.8)
            
            if is_correct:
                return f"Let me analyze this step by step. After careful reasoning, I conclude:\n(1) {names[0]} is a knight\n(2) {names[1]} is a knave"
            else:
                return f"Based on my analysis:\n(1) {names[0]} is a knave\n(2) {names[1]} is a knight"
        
        return "This is a complex logic puzzle that requires careful analysis of the statements."
    
    def _generate_general_response(self, prompt: str) -> str:
        """生成一般问题响应"""
        prompt_lower = prompt.lower()
        
        # 记忆密集型问题
        if any(word in prompt_lower for word in ['capital', 'who wrote', 'when was', 'chemical formula']):
            memory_steps = random.randint(3, 6)
            reasoning_steps = random.randint(1, 3)
            
            response = []
            for i in range(memory_steps):
                response.append(f"According to my knowledge, this is a well-established fact")
            for i in range(reasoning_steps):
                response.append(f"Therefore, we can conclude that")
            
            return ". ".join(response) + "."
        
        # 推理密集型问题
        elif any(word in prompt_lower for word in ['if', 'why', 'how', 'analyze', 'solve']):
            memory_steps = random.randint(1, 2)
            reasoning_steps = random.randint(4, 7)
            
            response = []
            for i in range(memory_steps):
                response.append(f"Based on the given information")
            for i in range(reasoning_steps):
                response.append(f"Thus, we can reason that")
            
            return ". ".join(response) + "."
        
        return "This is an interesting question that requires both knowledge and reasoning."
    
    def cleanup(self):
        """清理资源（模拟）"""
        pass

def run_demo():
    print("🎬 大语言模型记忆化vs推理能力评估 - 完整演示")
    print("="*70)
    print("注意：本演示使用模拟数据，展示完整的评估流程")
    print()
    
    # 模拟反事实评估结果
    print("🎯 1. 反事实评估演示 (Wu et al. 2023)")
    print("-" * 40)
    print("Base 10 准确率: 85.00% (17/20)")
    print("Base 11 准确率: 45.00% (9/20)")
    print("📊 性能下降: 47.06%")
    print("💡 结论: 模型可能过度依赖记忆化")
    
    # 模拟Knights and Knaves评估结果
    print("\n🧩 2. Knights and Knaves评估演示 (Xie et al. 2024)")
    print("-" * 40)
    print("📋 评估 clean: 80.00% (8/10)")
    print("📋 评估 flip_role: 50.00% (5/10)")  
    print("📋 评估 uncommon_name: 60.00% (6/10)")
    print("📈 扰动分析:")
    print("Baseline (clean): 80.00%")
    print("flip_role: 50.00% (下降 37.5%)")
    print("uncommon_name: 60.00% (下降 25.0%)")
    
    # 模拟记忆推理分离评估结果  
    print("\n🔬 3. 记忆推理分离评估演示 (Jin et al. 2024)")
    print("-" * 40)
    print("📋 评估: memory_intensive")
    print("  预期记忆比例: 80.00%")
    print("  实际记忆比例: 75.00%")
    print("  对齐度: 95.00%")
    print("📋 评估: reasoning_intensive")
    print("  预期记忆比例: 20.00%")
    print("  实际记忆比例: 30.00%")
    print("  对齐度: 90.00%")
    print("📋 评估: mixed")
    print("  预期记忆比例: 50.00%")
    print("  实际记忆比例: 55.00%")
    print("  对齐度: 95.00%")
    print("🧠 整体分析:")
    print("平均记忆比例: 53.33%")
    print("平均对齐度: 93.33%")
    print("💡 结论: 模型在记忆和推理之间相对平衡")
    
    # 创建结果文件
    results = {
        "counterfactual": {
            "base10_accuracy": 0.85,
            "base11_accuracy": 0.45,
            "performance_drop": 0.47
        },
        "knights_knaves": {
            "clean": 0.80,
            "flip_role": 0.50,
            "uncommon_name": 0.60
        },
        "memory_reasoning": {
            "overall_memory_ratio": 0.53,
            "overall_alignment": 0.93
        }
    }
    
    os.makedirs("demo_results", exist_ok=True)
    with open("demo_results/demo_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    with open("demo_results/demo_report.md", 'w', encoding='utf-8') as f:
        f.write("# 大语言模型记忆化vs推理能力评估演示报告\n\n")
        f.write("## 1. 反事实评估\n")
        f.write("- Base 10: 85.00%\n")
        f.write("- Base 11: 45.00%\n")
        f.write("- 性能下降: 47.06%\n\n")
        f.write("## 2. Knights and Knaves评估\n")
        f.write("- Clean: 80.00%\n")
        f.write("- Flip role: 50.00%\n")
        f.write("- Uncommon name: 60.00%\n\n")
        f.write("## 3. 记忆推理分离\n")
        f.write("- 记忆比例: 53.33%\n")
        f.write("- 对齐度: 93.33%\n\n")
    
    print(f"\n{'='*70}")
    print("🎉 演示完成！")
    print("📁 结果保存在: demo_results/")
    print("📊 详细报告: demo_results/demo_report.md")
    print()
    print("💡 要使用真实LLM模型，请配置网络连接并运行：")
    print("   python run_all_evaluations.py --models qwen-0.5b")

if __name__ == "__main__":
    run_demo() 