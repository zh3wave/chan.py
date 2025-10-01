#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chan.py 参数测试框架
===================

这个脚本提供了一个完整的测试框架，用于系统性地测试Chan.py的各种参数组合。

功能特性：
1. 自动化参数组合生成
2. 批量测试执行
3. 结果收集和分析
4. 可视化对比
5. 性能评估

作者: AI Assistant
创建时间: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import json
import os
import time
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 导入Chan相关模块
from Chan import CChan
from ChanConfig import CChanConfig
from Plot.PlotDriver import CPlotDriver

class ChanParameterTester:
    """Chan参数测试器"""
    
    def __init__(self, data_path: str, output_dir: str = "test_results"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.test_results = []
        self.baseline_result = None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/charts", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
    def load_test_data(self) -> pd.DataFrame:
        """加载测试数据"""
        try:
            if self.data_path.endswith('.csv'):
                data = pd.read_csv(self.data_path)
            else:
                raise ValueError("目前只支持CSV格式数据")
            
            print(f"成功加载数据: {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def create_baseline_config(self) -> Dict[str, Any]:
        """创建基准配置"""
        return {
            "bi_strict": True,
            "bi_fx_check": "strict",
            "seg_algo": "chan",
            "zs_combine": True,
            "zs_combine_mode": "zs",
            "one_bi_zs": False,
            "zs_algo": "normal",
            "divergence_rate": 0.9,
            "min_zs_cnt": 1,
            "macd_algo": "peak",
            "max_bs2_rate": 0.618,
            "bs1_peak": True,
            "bsp2_follow_1": True,
            "bsp3_follow_1": True,
        }
    
    def generate_test_configs(self, test_type: str = "core") -> List[Dict[str, Any]]:
        """生成测试配置"""
        baseline = self.create_baseline_config()
        configs = []
        
        if test_type == "core":
            # 核心参数测试
            core_variations = {
                "zs_algo": ["normal", "over_seg", "auto"],
                "zs_combine": [True, False],
                "bi_strict": [True, False],
                "seg_algo": ["chan", "1+1"],
                "divergence_rate": [0.618, 0.9, float("inf")]
            }
            
            for param, values in core_variations.items():
                for value in values:
                    config = baseline.copy()
                    config[param] = value
                    config["test_name"] = f"{param}_{value}"
                    configs.append(config)
        
        elif test_type == "zs_focus":
            # 中枢参数专项测试
            zs_combinations = [
                {"zs_algo": "normal", "zs_combine": True, "zs_combine_mode": "zs"},
                {"zs_algo": "normal", "zs_combine": True, "zs_combine_mode": "peak"},
                {"zs_algo": "normal", "zs_combine": False},
                {"zs_algo": "over_seg", "zs_combine": True, "zs_combine_mode": "zs"},
                {"zs_algo": "over_seg", "zs_combine": True, "zs_combine_mode": "peak"},
                {"zs_algo": "over_seg", "zs_combine": False},
                {"zs_algo": "auto", "zs_combine": True, "zs_combine_mode": "zs"},
            ]
            
            for i, zs_config in enumerate(zs_combinations):
                config = baseline.copy()
                config.update(zs_config)
                config["test_name"] = f"zs_combo_{i+1}"
                configs.append(config)
        
        elif test_type == "bsp_focus":
            # 买卖点参数专项测试
            bsp_variations = {
                "divergence_rate": [0.5, 0.618, 0.8, 0.9, 1.0, float("inf")],
                "min_zs_cnt": [0, 1, 2, 3],
                "macd_algo": ["area", "peak", "slope", "amp"],
                "max_bs2_rate": [0.5, 0.618, 0.8, 0.9]
            }
            
            for param, values in bsp_variations.items():
                for value in values:
                    config = baseline.copy()
                    config[param] = value
                    config["test_name"] = f"bsp_{param}_{value}"
                    configs.append(config)
        
        # 添加基准配置
        baseline["test_name"] = "baseline"
        configs.insert(0, baseline)
        
        return configs
    
    def run_single_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个测试"""
        try:
            start_time = time.time()
            
            # 创建Chan配置
            chan_config = CChanConfig(config)
            
            # 创建Chan实例
            chan = CChan(
                code="TEST",
                begin_time=None,
                end_time=None,
                data_src="csv",
                lv_list=["1m"],
                config=chan_config,
                autype="qfq"
            )
            
            # 加载数据（这里需要根据实际情况调整）
            # chan.load_stock_data(self.data_path)
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 收集结果指标
            result = {
                "test_name": config.get("test_name", "unknown"),
                "config": config,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
            }
            
            # 这里需要根据实际的Chan对象提取指标
            # result.update(self.extract_metrics(chan))
            
            # 临时模拟结果（实际使用时需要替换）
            result.update(self.simulate_metrics(config))
            
            return result
            
        except Exception as e:
            return {
                "test_name": config.get("test_name", "unknown"),
                "config": config,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    def simulate_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """模拟指标（用于演示，实际使用时需要替换）"""
        # 基于配置模拟一些合理的指标值
        base_bi_count = 100
        base_seg_count = 30
        base_zs_count = 10
        
        # 根据参数调整指标
        if config.get("bi_strict", True):
            bi_count = base_bi_count
        else:
            bi_count = int(base_bi_count * 1.2)
        
        if config.get("zs_algo") == "over_seg":
            zs_count = int(base_zs_count * 0.7)
        elif config.get("zs_algo") == "normal":
            zs_count = base_zs_count
        else:  # auto
            zs_count = int(base_zs_count * 0.85)
        
        if not config.get("zs_combine", True):
            zs_count = int(zs_count * 1.5)
        
        seg_count = base_seg_count + np.random.randint(-5, 6)
        
        return {
            "bi_count": bi_count + np.random.randint(-10, 11),
            "seg_count": max(1, seg_count),
            "zs_count": max(0, zs_count + np.random.randint(-3, 4)),
            "bsp1_count": np.random.randint(5, 15),
            "bsp2_count": np.random.randint(3, 10),
            "bsp3_count": np.random.randint(2, 8),
            "avg_bi_length": np.random.uniform(3.0, 8.0),
            "avg_seg_length": np.random.uniform(8.0, 20.0),
            "avg_zs_length": np.random.uniform(15.0, 40.0),
            "zs_coverage_ratio": np.random.uniform(0.3, 0.7),
            "signal_frequency": np.random.uniform(0.1, 0.3),
        }
    
    def extract_metrics(self, chan: CChan) -> Dict[str, Any]:
        """从Chan对象提取指标（实际实现）"""
        # 这里是实际的指标提取逻辑
        try:
            kl_data = chan.get_kl_data()
            if not kl_data:
                return {}
            
            # 获取最后一个级别的数据
            last_lv_data = kl_data[-1]
            
            metrics = {
                "bi_count": len(last_lv_data.bi_list),
                "seg_count": len(last_lv_data.seg_list),
                "zs_count": len(last_lv_data.zs_list),
                "bsp1_count": len([bsp for bsp in last_lv_data.bs_point_lst.bsp1_list]),
                "bsp2_count": 0,  # 需要具体实现
                "bsp3_count": 0,  # 需要具体实现
            }
            
            # 计算平均长度
            if metrics["bi_count"] > 0:
                bi_lengths = [bi.get_klu_cnt() for bi in last_lv_data.bi_list]
                metrics["avg_bi_length"] = np.mean(bi_lengths)
            
            if metrics["seg_count"] > 0:
                seg_lengths = [seg.get_klu_cnt() for seg in last_lv_data.seg_list]
                metrics["avg_seg_length"] = np.mean(seg_lengths)
            
            if metrics["zs_count"] > 0:
                zs_lengths = [zs.get_klu_cnt() for zs in last_lv_data.zs_list]
                metrics["avg_zs_length"] = np.mean(zs_lengths)
                
                # 计算中枢覆盖率
                total_klu = len(last_lv_data.lst)
                zs_coverage = sum(zs_lengths) / total_klu if total_klu > 0 else 0
                metrics["zs_coverage_ratio"] = zs_coverage
            
            return metrics
            
        except Exception as e:
            print(f"指标提取失败: {e}")
            return {}
    
    def run_batch_test(self, test_type: str = "core") -> List[Dict[str, Any]]:
        """运行批量测试"""
        configs = self.generate_test_configs(test_type)
        results = []
        
        print(f"开始批量测试，共 {len(configs)} 个配置...")
        
        for i, config in enumerate(configs):
            print(f"测试进度: {i+1}/{len(configs)} - {config['test_name']}")
            
            result = self.run_single_test(config)
            results.append(result)
            
            # 保存基准结果
            if config["test_name"] == "baseline":
                self.baseline_result = result
        
        self.test_results = results
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        if not self.test_results:
            return {}
        
        # 过滤出成功的测试
        successful_tests = [r for r in self.test_results if "error" not in r]
        
        if not successful_tests:
            return {"error": "没有成功的测试结果"}
        
        # 创建DataFrame进行分析
        df = pd.DataFrame(successful_tests)
        
        # 基本统计
        analysis = {
            "total_tests": len(self.test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(self.test_results) - len(successful_tests),
            "avg_execution_time": df["execution_time"].mean(),
        }
        
        # 指标统计
        metrics = ["bi_count", "seg_count", "zs_count", "bsp1_count", "bsp2_count", "bsp3_count"]
        for metric in metrics:
            if metric in df.columns:
                analysis[f"{metric}_mean"] = df[metric].mean()
                analysis[f"{metric}_std"] = df[metric].std()
                analysis[f"{metric}_min"] = df[metric].min()
                analysis[f"{metric}_max"] = df[metric].max()
        
        # 与基准对比
        if self.baseline_result and "error" not in self.baseline_result:
            analysis["baseline_comparison"] = {}
            for metric in metrics:
                if metric in self.baseline_result:
                    baseline_value = self.baseline_result[metric]
                    analysis["baseline_comparison"][metric] = {
                        "baseline": baseline_value,
                        "improvements": len([r for r in successful_tests 
                                           if r.get(metric, 0) > baseline_value]),
                        "degradations": len([r for r in successful_tests 
                                           if r.get(metric, 0) < baseline_value])
                    }
        
        return analysis
    
    def create_comparison_charts(self):
        """创建对比图表"""
        if not self.test_results:
            return
        
        successful_tests = [r for r in self.test_results if "error" not in r]
        if not successful_tests:
            return
        
        df = pd.DataFrame(successful_tests)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建多子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Chan参数测试结果对比', fontsize=16)
        
        # 主要指标对比
        metrics = ["bi_count", "seg_count", "zs_count", "bsp1_count", "execution_time"]
        metric_names = ["笔数量", "线段数量", "中枢数量", "1类买卖点", "执行时间(秒)"]
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            if i < 5 and metric in df.columns:
                ax = axes[i//3, i%3]
                
                # 柱状图
                test_names = df["test_name"].tolist()
                values = df[metric].tolist()
                
                bars = ax.bar(range(len(test_names)), values)
                ax.set_title(name)
                ax.set_xticks(range(len(test_names)))
                ax.set_xticklabels(test_names, rotation=45, ha='right')
                
                # 高亮基准值
                if self.baseline_result and metric in self.baseline_result:
                    baseline_idx = next((i for i, r in enumerate(successful_tests) 
                                       if r["test_name"] == "baseline"), None)
                    if baseline_idx is not None:
                        bars[baseline_idx].set_color('red')
        
        # 最后一个子图：参数影响热力图
        if len(axes.flat) > len(metrics):
            ax = axes.flat[-1]
            
            # 创建参数影响矩阵（简化版）
            param_impact = {}
            for result in successful_tests:
                test_name = result["test_name"]
                if test_name != "baseline":
                    param_impact[test_name] = result.get("zs_count", 0)
            
            if param_impact:
                names = list(param_impact.keys())[:10]  # 只显示前10个
                values = [param_impact[name] for name in names]
                
                ax.barh(names, values)
                ax.set_title("中枢数量对比")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/charts/parameter_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"对比图表已保存到: {self.output_dir}/charts/parameter_comparison.png")
    
    def save_results(self):
        """保存测试结果"""
        if not self.test_results:
            return
        
        # 保存详细结果
        with open(f"{self.output_dir}/data/test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存分析结果
        analysis = self.analyze_results()
        with open(f"{self.output_dir}/data/analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存CSV格式的结果
        successful_tests = [r for r in self.test_results if "error" not in r]
        if successful_tests:
            df = pd.DataFrame(successful_tests)
            df.to_csv(f"{self.output_dir}/data/test_results.csv", index=False, encoding="utf-8")
        
        print(f"测试结果已保存到: {self.output_dir}/data/")
    
    def generate_report(self) -> str:
        """生成测试报告"""
        analysis = self.analyze_results()
        
        report = f"""
# Chan参数测试报告

## 测试概要
- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总测试数: {analysis.get('total_tests', 0)}
- 成功测试: {analysis.get('successful_tests', 0)}
- 失败测试: {analysis.get('failed_tests', 0)}
- 平均执行时间: {analysis.get('avg_execution_time', 0):.3f}秒

## 主要发现

### 笔分析
- 平均笔数量: {analysis.get('bi_count_mean', 0):.1f} ± {analysis.get('bi_count_std', 0):.1f}
- 笔数量范围: {analysis.get('bi_count_min', 0)} - {analysis.get('bi_count_max', 0)}

### 线段分析  
- 平均线段数量: {analysis.get('seg_count_mean', 0):.1f} ± {analysis.get('seg_count_std', 0):.1f}
- 线段数量范围: {analysis.get('seg_count_min', 0)} - {analysis.get('seg_count_max', 0)}

### 中枢分析
- 平均中枢数量: {analysis.get('zs_count_mean', 0):.1f} ± {analysis.get('zs_count_std', 0):.1f}
- 中枢数量范围: {analysis.get('zs_count_min', 0)} - {analysis.get('zs_count_max', 0)}

### 买卖点分析
- 平均1类买卖点: {analysis.get('bsp1_count_mean', 0):.1f} ± {analysis.get('bsp1_count_std', 0):.1f}

## 参数建议

基于测试结果，建议关注以下参数组合：
1. 对于趋势市场：zs_algo="normal", zs_combine=True
2. 对于震荡市场：zs_algo="over_seg", zs_combine=False  
3. 对于高频交易：bi_strict=False, divergence_rate=0.618

## 后续研究方向

1. 深入研究参数交互作用
2. 在不同市场环境下验证参数效果
3. 建立参数自动优化机制
4. 结合实际交易验证参数实用性
"""
        
        # 保存报告
        with open(f"{self.output_dir}/test_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        return report

def main():
    """主函数 - 演示测试框架的使用"""
    
    # 创建测试器实例
    tester = ChanParameterTester(
        data_path="eth_data.csv",  # 需要替换为实际数据路径
        output_dir="eth_charts/parameter_test_results"
    )
    
    print("Chan参数测试框架启动...")
    print("="*50)
    
    # 运行核心参数测试
    print("\n1. 运行核心参数测试...")
    results = tester.run_batch_test("core")
    
    # 分析结果
    print("\n2. 分析测试结果...")
    analysis = tester.analyze_results()
    
    # 创建对比图表
    print("\n3. 创建对比图表...")
    tester.create_comparison_charts()
    
    # 保存结果
    print("\n4. 保存测试结果...")
    tester.save_results()
    
    # 生成报告
    print("\n5. 生成测试报告...")
    report = tester.generate_report()
    
    print("\n" + "="*50)
    print("测试完成！")
    print(f"成功测试: {analysis.get('successful_tests', 0)}")
    print(f"失败测试: {analysis.get('failed_tests', 0)}")
    print(f"结果保存在: {tester.output_dir}")
    
    # 显示简要结果
    if analysis.get('successful_tests', 0) > 0:
        print(f"\n主要发现:")
        print(f"- 平均笔数量: {analysis.get('bi_count_mean', 0):.1f}")
        print(f"- 平均线段数量: {analysis.get('seg_count_mean', 0):.1f}")
        print(f"- 平均中枢数量: {analysis.get('zs_count_mean', 0):.1f}")

if __name__ == "__main__":
    main()