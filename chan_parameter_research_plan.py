#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chan.py 参数研究计划
===================

这个脚本提供了一个系统性的研究框架，用于探索Chan.py中各种参数设置的可能性及其结果。

研究目标：
1. 识别关键参数对缠论分析结果的影响
2. 找到最优参数组合
3. 理解不同参数设置的适用场景
4. 建立参数调优的方法论

作者: AI Assistant
创建时间: 2024
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Any, Tuple
import json
import os
from datetime import datetime

class ChanParameterResearcher:
    """Chan.py参数研究器"""
    
    def __init__(self, data_path: str, output_dir: str = "research_results"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.results = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """定义参数空间"""
        return {
            # 笔参数
            "bi_algo": ["normal", "fx"],
            "bi_strict": [True, False],
            "bi_fx_check": ["strict", "loss", "half", "totally"],
            "gap_as_kl": [True, False],
            "bi_end_is_peak": [True, False],
            "bi_allow_sub_peak": [True, False],
            
            # 线段参数
            "seg_algo": ["chan", "1+1", "break"],
            "left_seg_method": ["peak", "all"],
            
            # 中枢参数
            "zs_combine": [True, False],
            "zs_combine_mode": ["zs", "peak"],
            "one_bi_zs": [True, False],
            "zs_algo": ["normal", "over_seg", "auto"],
            
            # 买卖点参数
            "divergence_rate": [0.5, 0.618, 0.8, 0.9, 1.0, float("inf")],
            "min_zs_cnt": [0, 1, 2, 3],
            "bsp1_only_multibi_zs": [True, False],
            "max_bs2_rate": [0.5, 0.618, 0.8, 0.9, 0.9999],
            "macd_algo": ["area", "peak", "full_area", "diff", "slope", "amp"],
            "bs1_peak": [True, False],
            "bsp2_follow_1": [True, False],
            "bsp3_follow_1": [True, False],
            "bsp3_peak": [True, False],
            "strict_bsp3": [True, False],
        }
    
    def get_core_parameter_combinations(self) -> List[Dict[str, Any]]:
        """获取核心参数组合（减少组合数量）"""
        core_params = {
            "zs_algo": ["normal", "over_seg"],
            "zs_combine": [True, False],
            "zs_combine_mode": ["zs", "peak"],
            "bi_strict": [True, False],
            "seg_algo": ["chan", "1+1"],
            "macd_algo": ["peak", "area", "slope"],
            "divergence_rate": [0.618, 0.9, float("inf")],
        }
        
        combinations = []
        for values in product(*core_params.values()):
            combo = dict(zip(core_params.keys(), values))
            combinations.append(combo)
        
        return combinations
    
    def get_evaluation_metrics(self) -> List[str]:
        """定义评估指标"""
        return [
            "bi_count",           # 笔数量
            "seg_count",          # 线段数量
            "zs_count",           # 中枢数量
            "bsp1_count",         # 1类买卖点数量
            "bsp2_count",         # 2类买卖点数量
            "bsp3_count",         # 3类买卖点数量
            "avg_bi_length",      # 平均笔长度
            "avg_seg_length",     # 平均线段长度
            "avg_zs_length",      # 平均中枢长度
            "zs_coverage_ratio",  # 中枢覆盖率
            "trend_consistency",  # 趋势一致性
            "signal_frequency",   # 信号频率
            "computation_time",   # 计算时间
        ]
    
    def create_research_phases(self) -> Dict[str, Dict]:
        """创建研究阶段"""
        return {
            "phase1_core_exploration": {
                "description": "核心参数探索",
                "parameters": ["zs_algo", "zs_combine", "bi_strict", "seg_algo"],
                "focus": "基础结构参数对分析结果的影响",
                "sample_size": 50
            },
            
            "phase2_zs_deep_dive": {
                "description": "中枢参数深度研究",
                "parameters": ["zs_algo", "zs_combine", "zs_combine_mode", "one_bi_zs"],
                "focus": "中枢识别和合并策略的优化",
                "sample_size": 100
            },
            
            "phase3_bsp_optimization": {
                "description": "买卖点参数优化",
                "parameters": ["divergence_rate", "min_zs_cnt", "macd_algo", "max_bs2_rate"],
                "focus": "买卖点识别准确性和实用性",
                "sample_size": 150
            },
            
            "phase4_bi_seg_tuning": {
                "description": "笔段参数调优",
                "parameters": ["bi_fx_check", "bi_end_is_peak", "seg_algo", "left_seg_method"],
                "focus": "笔段识别精度和稳定性",
                "sample_size": 80
            },
            
            "phase5_comprehensive": {
                "description": "综合参数测试",
                "parameters": "all",
                "focus": "最优参数组合发现",
                "sample_size": 200
            }
        }
    
    def generate_test_scenarios(self) -> List[Dict]:
        """生成测试场景"""
        scenarios = [
            {
                "name": "trending_market",
                "description": "趋势市场",
                "data_filter": "high_volatility_trend",
                "expected_behavior": "清晰的笔段结构，较少中枢"
            },
            {
                "name": "sideways_market", 
                "description": "震荡市场",
                "data_filter": "low_volatility_range",
                "expected_behavior": "较多中枢，频繁买卖点"
            },
            {
                "name": "volatile_market",
                "description": "高波动市场", 
                "data_filter": "high_volatility_range",
                "expected_behavior": "复杂笔段结构，多层级中枢"
            },
            {
                "name": "low_volume_market",
                "description": "低成交量市场",
                "data_filter": "low_volume",
                "expected_behavior": "较少有效信号，需要宽松参数"
            }
        ]
        return scenarios
    
    def create_parameter_sensitivity_analysis(self) -> Dict:
        """创建参数敏感性分析计划"""
        return {
            "single_parameter_sweep": {
                "description": "单参数扫描",
                "method": "固定其他参数，单独变化一个参数",
                "parameters": ["divergence_rate", "min_zs_cnt", "max_bs2_rate"]
            },
            
            "parameter_interaction": {
                "description": "参数交互作用",
                "method": "两两参数组合分析",
                "key_pairs": [
                    ("zs_algo", "zs_combine"),
                    ("bi_strict", "bi_fx_check"),
                    ("seg_algo", "left_seg_method"),
                    ("macd_algo", "divergence_rate")
                ]
            },
            
            "stability_analysis": {
                "description": "参数稳定性分析",
                "method": "在不同数据段上测试相同参数",
                "focus": "参数设置的鲁棒性"
            }
        }
    
    def create_evaluation_framework(self) -> Dict:
        """创建评估框架"""
        return {
            "quantitative_metrics": {
                "structure_metrics": [
                    "bi_count", "seg_count", "zs_count",
                    "avg_bi_length", "avg_seg_length"
                ],
                "signal_metrics": [
                    "bsp1_count", "bsp2_count", "bsp3_count",
                    "signal_frequency", "signal_quality"
                ],
                "performance_metrics": [
                    "computation_time", "memory_usage"
                ]
            },
            
            "qualitative_assessment": {
                "visual_inspection": [
                    "chart_clarity", "structure_reasonableness",
                    "signal_timing", "trend_identification"
                ],
                "theoretical_consistency": [
                    "chan_theory_compliance", "logical_coherence"
                ]
            },
            
            "comparative_analysis": {
                "baseline_comparison": "与默认参数对比",
                "cross_validation": "不同时间段验证",
                "scenario_testing": "不同市场环境测试"
            }
        }
    
    def save_research_plan(self):
        """保存研究计划"""
        research_plan = {
            "title": "Chan.py 参数研究计划",
            "created_at": datetime.now().isoformat(),
            "parameter_space": self.get_parameter_space(),
            "core_combinations": len(self.get_core_parameter_combinations()),
            "evaluation_metrics": self.get_evaluation_metrics(),
            "research_phases": self.create_research_phases(),
            "test_scenarios": self.generate_test_scenarios(),
            "sensitivity_analysis": self.create_parameter_sensitivity_analysis(),
            "evaluation_framework": self.create_evaluation_framework()
        }
        
        with open(f"{self.output_dir}/research_plan.json", "w", encoding="utf-8") as f:
            json.dump(research_plan, f, ensure_ascii=False, indent=2)
        
        print(f"研究计划已保存到: {self.output_dir}/research_plan.json")
        return research_plan

def main():
    """主函数 - 演示研究计划的使用"""
    
    # 创建研究器实例
    researcher = ChanParameterResearcher(
        data_path="eth_data.csv",
        output_dir="eth_charts/research_results"
    )
    
    # 生成并保存研究计划
    plan = researcher.save_research_plan()
    
    # 打印研究概要
    print("\n" + "="*60)
    print("Chan.py 参数研究计划概要")
    print("="*60)
    
    print(f"\n📊 参数空间大小:")
    param_space = researcher.get_parameter_space()
    total_combinations = 1
    for param, values in param_space.items():
        total_combinations *= len(values)
        print(f"  {param}: {len(values)} 个选项")
    print(f"  总组合数: {total_combinations:,}")
    
    print(f"\n🎯 核心参数组合: {len(researcher.get_core_parameter_combinations())} 个")
    
    print(f"\n📈 评估指标: {len(researcher.get_evaluation_metrics())} 个")
    for metric in researcher.get_evaluation_metrics():
        print(f"  - {metric}")
    
    print(f"\n🔬 研究阶段: {len(researcher.create_research_phases())} 个")
    for phase_name, phase_info in researcher.create_research_phases().items():
        print(f"  {phase_name}: {phase_info['description']}")
    
    print(f"\n🎭 测试场景: {len(researcher.generate_test_scenarios())} 个")
    for scenario in researcher.generate_test_scenarios():
        print(f"  {scenario['name']}: {scenario['description']}")
    
    print("\n" + "="*60)
    print("建议的研究步骤:")
    print("="*60)
    print("1. 从 phase1_core_exploration 开始")
    print("2. 使用 ETH 数据进行初步测试")
    print("3. 重点关注 zs_algo 和 zs_combine 参数")
    print("4. 记录每个参数组合的结果")
    print("5. 逐步深入到其他研究阶段")
    print("6. 最终形成参数调优指南")

if __name__ == "__main__":
    main()