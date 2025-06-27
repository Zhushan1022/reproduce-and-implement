#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大语言模型记忆化vs推理能力评估 - 统一启动脚本

整合所有复现的论文方法，使用本地LLM模型进行全面评估
"""

import sys
import os
import time
import json
from typing import Dict, List, Optional
import argparse

# 导入本地LLM接口
from local_llm_interface import load_model, SUPPORTED_MODELS

# 导入各评估方法
sys.path.append('3/counterfactual-evaluation-master/')
from local_model_eval import run_comprehensive_evaluation as run_counterfactual_eval

sys.path.append('1/mem-kk-logic-main/')
from local_model_eval_kk import run_kk_evaluation

sys.path.append('4/Disentangling-Memory-and-Reasoning-main/')
from local_model_eval_jin import run_jin_evaluation

def test_model_availability(model_key: str) -> bool:
    """测试模型是否可用"""
    print(f"🔍 测试模型可用性: {model_key}")
    
    try:
        llm = load_model(model_key)
        
        # 简单测试
        test_prompt = "What is 2+2?"
        response = llm.query(test_prompt, max_new_tokens=50)
        
        llm.cleanup()
        
        if response and len(response.strip()) > 0:
            print(f"✅ 模型 {model_key} 可用")
            return True
        else:
            print(f"❌ 模型 {model_key} 无回应")
            return False
            
    except Exception as e:
        print(f"❌ 模型 {model_key} 不可用: {e}")
        return False

def run_single_evaluation(method: str, model_key: str) -> Optional[Dict]:
    """运行单个评估方法"""
    print(f"\n{'='*20} {method.upper()} 评估 {'='*20}")
    
    try:
        if method == "counterfactual":
            # Wu et al. (2023) 反事实评估
            print("📊 运行反事实评估 (Wu et al. 2023)")
            from local_model_eval import compare_bases
            return compare_bases(model_key, num_problems=20)
            
        elif method == "knights_knaves":
            # Xie et al. (2024) Knights and Knaves
            print("🧩 运行Knights and Knaves评估 (Xie et al. 2024)")
            from local_model_eval_kk import LocalKKEvaluator
            evaluator = LocalKKEvaluator(model_key)
            result = evaluator.compare_perturbations(limit=10)
            evaluator.cleanup_model()
            return result
            
        elif method == "memory_reasoning":
            # Jin et al. (2024) 记忆推理分离
            print("🔬 运行记忆推理分离评估 (Jin et al. 2024)")
            return run_jin_evaluation(model_key)
            
        else:
            print(f"❌ 未知评估方法: {method}")
            return None
            
    except Exception as e:
        print(f"❌ {method} 评估失败: {e}")
        return {"error": str(e)}

def run_comprehensive_evaluation(model_keys: List[str], 
                               methods: List[str],
                               output_dir: str = "evaluation_results") -> Dict:
    """运行全面评估"""
    print("🚀 启动大语言模型记忆化vs推理能力全面评估")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 总体结果
    all_results = {
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_tested": model_keys,
        "methods_used": methods,
        "results": {}
    }
    
    start_time = time.time()
    
    for model_key in model_keys:
        print(f"\n🎯 评估模型: {model_key}")
        print("-" * 50)
        
        # 测试模型可用性
        if not test_model_availability(model_key):
            all_results["results"][model_key] = {
                "status": "unavailable",
                "error": "Model not available"
            }
            continue
        
        # 运行各个评估方法
        model_results = {}
        
        for method in methods:
            print(f"\n📋 运行 {method} 评估...")
            
            method_start = time.time()
            result = run_single_evaluation(method, model_key)
            method_time = time.time() - method_start
            
            if result:
                result["evaluation_time"] = method_time
                model_results[method] = result
                print(f"✅ {method} 评估完成 ({method_time:.1f}秒)")
            else:
                model_results[method] = {
                    "error": "Evaluation failed",
                    "evaluation_time": method_time
                }
                print(f"❌ {method} 评估失败")
        
        all_results["results"][model_key] = model_results
        
        # 保存中间结果
        model_file = os.path.join(output_dir, f"{model_key}_results.json")
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)
        print(f"💾 {model_key} 结果已保存到: {model_file}")
    
    total_time = time.time() - start_time
    all_results["total_evaluation_time"] = total_time
    
    # 保存总体结果
    summary_file = os.path.join(output_dir, "comprehensive_results.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 生成分析报告
    generate_analysis_report(all_results, output_dir)
    
    print(f"\n{'='*70}")
    print("🎉 全面评估完成！")
    print(f"⏱️  总用时: {total_time:.1f}秒")
    print(f"📁 结果保存在: {output_dir}/")
    print(f"📊 分析报告: {output_dir}/analysis_report.md")
    
    return all_results

def generate_analysis_report(results: Dict, output_dir: str):
    """生成分析报告"""
    report_path = os.path.join(output_dir, "analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 大语言模型记忆化vs推理能力评估报告\n\n")
        f.write(f"**评估时间**: {results['evaluation_time']}\n\n")
        f.write(f"**总用时**: {results.get('total_evaluation_time', 0):.1f}秒\n\n")
        
        f.write("## 评估概览\n\n")
        f.write(f"- **测试模型**: {', '.join(results['models_tested'])}\n")
        f.write(f"- **评估方法**: {', '.join(results['methods_used'])}\n\n")
        
        f.write("## 详细结果\n\n")
        
        for model_key, model_results in results["results"].items():
            f.write(f"### {model_key}\n\n")
            
            if "status" in model_results and model_results["status"] == "unavailable":
                f.write("❌ **模型不可用**\n\n")
                continue
            
            for method, method_result in model_results.items():
                f.write(f"#### {method}\n\n")
                
                if "error" in method_result:
                    f.write(f"❌ **失败**: {method_result['error']}\n\n")
                    continue
                
                # 根据方法类型生成不同的报告
                if method == "counterfactual":
                    if "base10" in method_result and "base11" in method_result:
                        base10_acc = method_result["base10"].get("accuracy", 0)
                        base11_acc = method_result["base11"].get("accuracy", 0)
                        f.write(f"- **Base 10 准确率**: {base10_acc:.2%}\n")
                        f.write(f"- **Base 11 准确率**: {base11_acc:.2%}\n")
                        if base10_acc > 0:
                            drop = (base10_acc - base11_acc) / base10_acc
                            f.write(f"- **性能下降**: {drop:.2%}\n")
                
                elif method == "knights_knaves":
                    # 分析扰动结果
                    for perturb_type, perturb_result in method_result.items():
                        if isinstance(perturb_result, dict) and "accuracy" in perturb_result:
                            acc = perturb_result["accuracy"]
                            f.write(f"- **{perturb_type}**: {acc:.2%}\n")
                
                elif method == "memory_reasoning":
                    if "summary_analysis" in method_result:
                        summary = method_result["summary_analysis"]
                        f.write(f"- **整体记忆比例**: {summary.get('overall_memory_ratio', 0):.2%}\n")
                        f.write(f"- **整体对齐度**: {summary.get('overall_alignment', 0):.2%}\n")
                
                f.write("\n")
        
        f.write("## 结论\n\n")
        f.write("根据评估结果，可以得出以下结论：\n\n")
        f.write("1. **反事实评估**：检测模型在分布外任务上的性能下降\n")
        f.write("2. **逻辑扰动评估**：分析模型对不同类型扰动的敏感性\n")
        f.write("3. **记忆推理分离**：量化模型依赖记忆vs推理的程度\n\n")
        f.write("详细的数值结果请参考对应的JSON文件。\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="大语言模型记忆化vs推理能力评估")
    
    parser.add_argument("--models", nargs="+", 
                       default=["qwen-0.5b"],
                       choices=list(SUPPORTED_MODELS.keys()),
                       help="要测试的模型")
    
    parser.add_argument("--methods", nargs="+",
                       default=["counterfactual", "knights_knaves", "memory_reasoning"],
                       choices=["counterfactual", "knights_knaves", "memory_reasoning"],
                       help="要运行的评估方法")
    
    parser.add_argument("--output-dir", default="evaluation_results",
                       help="结果输出目录")
    
    parser.add_argument("--test-only", action="store_true",
                       help="仅测试模型可用性")
    
    args = parser.parse_args()
    
    print("🔬 大语言模型记忆化vs推理能力评估工具")
    print("="*50)
    
    # 显示可用模型
    print("📋 支持的模型:")
    for key, config in SUPPORTED_MODELS.items():
        print(f"  {key}: {config['model_name']}")
    print()
    
    if args.test_only:
        # 仅测试模型
        print("🔍 测试模型可用性...")
        for model_key in args.models:
            test_model_availability(model_key)
    else:
        # 运行完整评估
        try:
            results = run_comprehensive_evaluation(
                model_keys=args.models,
                methods=args.methods,
                output_dir=args.output_dir
            )
        except KeyboardInterrupt:
            print("\n❌ 用户中断评估")
        except Exception as e:
            print(f"\n❌ 评估过程出错: {e}")

if __name__ == "__main__":
    main() 