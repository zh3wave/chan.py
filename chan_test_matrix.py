"""
Chan.py 参数组合测试矩阵
用于系统性地测试不同参数组合的效果

Version: 1.0
Created: 2024
"""

import itertools
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import pandas as pd

@dataclass
class TestScenario:
    """测试场景定义"""
    name: str
    description: str
    config: Dict[str, Any]
    priority: str  # high, medium, low
    expected_behavior: str

class ChanTestMatrix:
    """Chan.py 参数测试矩阵生成器"""
    
    def __init__(self):
        self.test_scenarios = []
        self.parameter_space = self._define_parameter_space()
        self.core_combinations = self._define_core_combinations()
        
    def _define_parameter_space(self) -> Dict[str, List[Any]]:
        """定义完整的参数空间"""
        return {
            # 笔参数
            'bi_algo': ['normal', 'fx'],
            'bi_strict': [True, False],
            'bi_fx_check': ['strict', 'half', 'totally'],
            'gap_as_kl': [True, False],
            'bi_end_is_peak': [True, False],
            'bi_allow_sub_peak': [True, False],
            
            # 线段参数
            'seg_algo': ['chan', '1+1', 'break'],
            'left_seg_method': ['peak', 'high', 'low'],
            
            # 中枢参数
            'zs_algo': ['normal', 'over_seg', 'auto'],
            'need_combine': [True, False],
            'zs_combine_mode': ['zs', 'peak', 'all'],
            'one_bi_zs': [True, False],
            
            # 买卖点参数
            'divergence_rate': [0.8, 0.9, 1.0, 1.1, 1.2],
            'min_zs_cnt': [0, 1, 2, 3],
            'macd_algo': ['area', 'peak', 'full_area', 'diff', 'slope', 'amp'],
            'bs_type': ['1', '2', '3', '1,2', '1,3', '2,3', '1,2,3'],
            
            # 技术指标参数
            'macd_fast': [12, 8, 10, 15],
            'macd_slow': [26, 20, 24, 30],
            'macd_signal': [9, 6, 8, 12],
            'rsi_period': [14, 10, 12, 16, 20],
        }
    
    def _define_core_combinations(self) -> List[Dict[str, Any]]:
        """定义核心参数组合"""
        return [
            # 保守配置
            {
                'name': 'conservative',
                'bi_algo': 'normal',
                'bi_strict': True,
                'seg_algo': 'chan',
                'zs_algo': 'normal',
                'divergence_rate': 0.9,
                'min_zs_cnt': 1,
                'macd_algo': 'area'
            },
            
            # 激进配置
            {
                'name': 'aggressive',
                'bi_algo': 'fx',
                'bi_strict': False,
                'seg_algo': 'break',
                'zs_algo': 'over_seg',
                'divergence_rate': 1.1,
                'min_zs_cnt': 0,
                'macd_algo': 'peak'
            },
            
            # 平衡配置
            {
                'name': 'balanced',
                'bi_algo': 'normal',
                'bi_strict': False,
                'seg_algo': '1+1',
                'zs_algo': 'auto',
                'divergence_rate': 1.0,
                'min_zs_cnt': 1,
                'macd_algo': 'diff'
            },
            
            # 高精度配置
            {
                'name': 'high_precision',
                'bi_algo': 'normal',
                'bi_strict': True,
                'bi_fx_check': 'strict',
                'seg_algo': 'chan',
                'zs_algo': 'normal',
                'need_combine': True,
                'divergence_rate': 0.8,
                'min_zs_cnt': 2,
                'macd_algo': 'full_area'
            },
            
            # 高敏感度配置
            {
                'name': 'high_sensitivity',
                'bi_algo': 'fx',
                'bi_strict': False,
                'bi_allow_sub_peak': True,
                'seg_algo': 'break',
                'zs_algo': 'over_seg',
                'one_bi_zs': True,
                'divergence_rate': 1.2,
                'min_zs_cnt': 0,
                'macd_algo': 'slope'
            }
        ]
    
    def generate_bi_focused_tests(self) -> List[TestScenario]:
        """生成笔参数专项测试"""
        scenarios = []
        
        # 笔算法对比
        for bi_algo in ['normal', 'fx']:
            for bi_strict in [True, False]:
                config = {
                    'bi_algo': bi_algo,
                    'bi_strict': bi_strict,
                    'seg_algo': 'chan',  # 固定其他参数
                    'zs_algo': 'normal',
                    'divergence_rate': 1.0,
                    'min_zs_cnt': 1,
                    'macd_algo': 'area'
                }
                
                scenarios.append(TestScenario(
                    name=f"bi_test_{bi_algo}_{bi_strict}",
                    description=f"笔算法测试: {bi_algo}, 严格模式: {bi_strict}",
                    config=config,
                    priority="high",
                    expected_behavior=f"{'更严格的笔识别' if bi_strict else '更宽松的笔识别'}"
                ))
        
        # 笔检查模式测试
        for fx_check in ['strict', 'half', 'totally']:
            config = {
                'bi_algo': 'normal',
                'bi_strict': True,
                'bi_fx_check': fx_check,
                'seg_algo': 'chan',
                'zs_algo': 'normal',
                'divergence_rate': 1.0,
                'min_zs_cnt': 1,
                'macd_algo': 'area'
            }
            
            scenarios.append(TestScenario(
                name=f"bi_fx_check_{fx_check}",
                description=f"笔分型检查模式: {fx_check}",
                config=config,
                priority="medium",
                expected_behavior=f"分型检查{fx_check}模式的笔识别效果"
            ))
        
        return scenarios
    
    def generate_seg_focused_tests(self) -> List[TestScenario]:
        """生成线段参数专项测试"""
        scenarios = []
        
        # 线段算法对比
        for seg_algo in ['chan', '1+1', 'break']:
            for left_method in ['peak', 'high', 'low']:
                config = {
                    'bi_algo': 'normal',
                    'bi_strict': True,
                    'seg_algo': seg_algo,
                    'left_seg_method': left_method,
                    'zs_algo': 'normal',
                    'divergence_rate': 1.0,
                    'min_zs_cnt': 1,
                    'macd_algo': 'area'
                }
                
                scenarios.append(TestScenario(
                    name=f"seg_test_{seg_algo}_{left_method}",
                    description=f"线段算法测试: {seg_algo}, 左侧方法: {left_method}",
                    config=config,
                    priority="high",
                    expected_behavior=f"线段算法{seg_algo}与左侧方法{left_method}的组合效果"
                ))
        
        return scenarios
    
    def generate_zs_focused_tests(self) -> List[TestScenario]:
        """生成中枢参数专项测试"""
        scenarios = []
        
        # 中枢算法对比
        for zs_algo in ['normal', 'over_seg', 'auto']:
            for need_combine in [True, False]:
                config = {
                    'bi_algo': 'normal',
                    'bi_strict': True,
                    'seg_algo': 'chan',
                    'zs_algo': zs_algo,
                    'need_combine': need_combine,
                    'divergence_rate': 1.0,
                    'min_zs_cnt': 1,
                    'macd_algo': 'area'
                }
                
                scenarios.append(TestScenario(
                    name=f"zs_test_{zs_algo}_{need_combine}",
                    description=f"中枢算法测试: {zs_algo}, 合并: {need_combine}",
                    config=config,
                    priority="high",
                    expected_behavior=f"中枢算法{zs_algo}{'合并' if need_combine else '不合并'}的效果"
                ))
        
        # 中枢合并模式测试
        for combine_mode in ['zs', 'peak', 'all']:
            config = {
                'bi_algo': 'normal',
                'bi_strict': True,
                'seg_algo': 'chan',
                'zs_algo': 'normal',
                'need_combine': True,
                'zs_combine_mode': combine_mode,
                'divergence_rate': 1.0,
                'min_zs_cnt': 1,
                'macd_algo': 'area'
            }
            
            scenarios.append(TestScenario(
                name=f"zs_combine_{combine_mode}",
                description=f"中枢合并模式测试: {combine_mode}",
                config=config,
                priority="medium",
                expected_behavior=f"中枢合并模式{combine_mode}的效果"
            ))
        
        return scenarios
    
    def generate_bsp_focused_tests(self) -> List[TestScenario]:
        """生成买卖点参数专项测试"""
        scenarios = []
        
        # 背驰率测试
        for div_rate in [0.8, 0.9, 1.0, 1.1, 1.2]:
            config = {
                'bi_algo': 'normal',
                'bi_strict': True,
                'seg_algo': 'chan',
                'zs_algo': 'normal',
                'divergence_rate': div_rate,
                'min_zs_cnt': 1,
                'macd_algo': 'area'
            }
            
            scenarios.append(TestScenario(
                name=f"bsp_div_rate_{div_rate}",
                description=f"背驰率测试: {div_rate}",
                config=config,
                priority="high",
                expected_behavior=f"背驰率{div_rate}的买卖点识别效果"
            ))
        
        # MACD算法测试
        for macd_algo in ['area', 'peak', 'full_area', 'diff', 'slope', 'amp']:
            config = {
                'bi_algo': 'normal',
                'bi_strict': True,
                'seg_algo': 'chan',
                'zs_algo': 'normal',
                'divergence_rate': 1.0,
                'min_zs_cnt': 1,
                'macd_algo': macd_algo
            }
            
            scenarios.append(TestScenario(
                name=f"bsp_macd_{macd_algo}",
                description=f"MACD算法测试: {macd_algo}",
                config=config,
                priority="high",
                expected_behavior=f"MACD算法{macd_algo}的背驰判断效果"
            ))
        
        return scenarios
    
    def generate_combination_tests(self) -> List[TestScenario]:
        """生成组合参数测试"""
        scenarios = []
        
        # 基于核心组合的变体测试
        for core_config in self.core_combinations:
            base_config = core_config.copy()
            name = base_config.pop('name')
            
            scenarios.append(TestScenario(
                name=f"combo_{name}",
                description=f"核心组合测试: {name}",
                config=base_config,
                priority="high",
                expected_behavior=f"{name}配置的综合效果"
            ))
        
        # 对比测试组合
        comparison_pairs = [
            ('conservative_vs_aggressive', 0, 1),
            ('conservative_vs_balanced', 0, 2),
            ('aggressive_vs_balanced', 1, 2),
        ]
        
        for pair_name, idx1, idx2 in comparison_pairs:
            config1 = self.core_combinations[idx1].copy()
            config2 = self.core_combinations[idx2].copy()
            
            scenarios.append(TestScenario(
                name=f"compare_{pair_name}",
                description=f"对比测试: {config1['name']} vs {config2['name']}",
                config={'config1': config1, 'config2': config2},
                priority="medium",
                expected_behavior=f"两种配置的对比效果分析"
            ))
        
        return scenarios
    
    def generate_sensitivity_tests(self) -> List[TestScenario]:
        """生成参数敏感性测试"""
        scenarios = []
        
        # 单参数敏感性测试
        sensitive_params = {
            'divergence_rate': [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            'min_zs_cnt': [0, 1, 2, 3, 4],
            'macd_fast': [8, 10, 12, 15, 18],
            'macd_slow': [20, 24, 26, 30, 35],
        }
        
        base_config = {
            'bi_algo': 'normal',
            'bi_strict': True,
            'seg_algo': 'chan',
            'zs_algo': 'normal',
            'divergence_rate': 1.0,
            'min_zs_cnt': 1,
            'macd_algo': 'area',
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
        for param_name, param_values in sensitive_params.items():
            for value in param_values:
                config = base_config.copy()
                config[param_name] = value
                
                scenarios.append(TestScenario(
                    name=f"sensitivity_{param_name}_{value}",
                    description=f"参数敏感性测试: {param_name} = {value}",
                    config=config,
                    priority="medium",
                    expected_behavior=f"参数{param_name}变化对结果的影响"
                ))
        
        return scenarios
    
    def generate_all_test_scenarios(self) -> List[TestScenario]:
        """生成所有测试场景"""
        all_scenarios = []
        
        # 添加各类专项测试
        all_scenarios.extend(self.generate_bi_focused_tests())
        all_scenarios.extend(self.generate_seg_focused_tests())
        all_scenarios.extend(self.generate_zs_focused_tests())
        all_scenarios.extend(self.generate_bsp_focused_tests())
        all_scenarios.extend(self.generate_combination_tests())
        all_scenarios.extend(self.generate_sensitivity_tests())
        
        return all_scenarios
    
    def create_test_matrix_df(self) -> pd.DataFrame:
        """创建测试矩阵DataFrame"""
        scenarios = self.generate_all_test_scenarios()
        
        data = []
        for scenario in scenarios:
            row = {
                'test_name': scenario.name,
                'description': scenario.description,
                'priority': scenario.priority,
                'expected_behavior': scenario.expected_behavior,
            }
            
            # 添加配置参数
            if isinstance(scenario.config, dict) and 'config1' not in scenario.config:
                row.update(scenario.config)
            else:
                row['config'] = json.dumps(scenario.config, ensure_ascii=False)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_test_matrix(self, filename: str = 'chan_test_matrix.xlsx'):
        """保存测试矩阵到Excel文件"""
        df = self.create_test_matrix_df()
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 主测试矩阵
            df.to_excel(writer, sheet_name='Test_Matrix', index=False)
            
            # 参数空间
            param_df = pd.DataFrame([
                {'parameter': k, 'possible_values': str(v)} 
                for k, v in self.parameter_space.items()
            ])
            param_df.to_excel(writer, sheet_name='Parameter_Space', index=False)
            
            # 核心组合
            core_df = pd.DataFrame(self.core_combinations)
            core_df.to_excel(writer, sheet_name='Core_Combinations', index=False)
        
        print(f"测试矩阵已保存到: {filename}")
        print(f"总测试场景数: {len(df)}")
        print(f"高优先级测试: {len(df[df['priority'] == 'high'])}")
        print(f"中优先级测试: {len(df[df['priority'] == 'medium'])}")
        print(f"低优先级测试: {len(df[df['priority'] == 'low'])}")
    
    def get_test_plan_summary(self) -> Dict[str, Any]:
        """获取测试计划摘要"""
        scenarios = self.generate_all_test_scenarios()
        
        summary = {
            'total_scenarios': len(scenarios),
            'by_priority': {
                'high': len([s for s in scenarios if s.priority == 'high']),
                'medium': len([s for s in scenarios if s.priority == 'medium']),
                'low': len([s for s in scenarios if s.priority == 'low']),
            },
            'by_category': {
                'bi_tests': len([s for s in scenarios if s.name.startswith('bi_')]),
                'seg_tests': len([s for s in scenarios if s.name.startswith('seg_')]),
                'zs_tests': len([s for s in scenarios if s.name.startswith('zs_')]),
                'bsp_tests': len([s for s in scenarios if s.name.startswith('bsp_')]),
                'combo_tests': len([s for s in scenarios if s.name.startswith('combo_')]),
                'sensitivity_tests': len([s for s in scenarios if s.name.startswith('sensitivity_')]),
                'compare_tests': len([s for s in scenarios if s.name.startswith('compare_')]),
            },
            'estimated_time': {
                'high_priority': len([s for s in scenarios if s.priority == 'high']) * 5,  # 5分钟每个
                'medium_priority': len([s for s in scenarios if s.priority == 'medium']) * 3,  # 3分钟每个
                'low_priority': len([s for s in scenarios if s.priority == 'low']) * 2,  # 2分钟每个
            }
        }
        
        summary['estimated_time']['total_minutes'] = sum(summary['estimated_time'].values())
        summary['estimated_time']['total_hours'] = summary['estimated_time']['total_minutes'] / 60
        
        return summary

def main():
    """主函数 - 演示测试矩阵生成"""
    print("=== Chan.py 参数测试矩阵生成器 ===\n")
    
    # 创建测试矩阵
    matrix = ChanTestMatrix()
    
    # 获取测试计划摘要
    summary = matrix.get_test_plan_summary()
    
    print("测试计划摘要:")
    print(f"总测试场景数: {summary['total_scenarios']}")
    print(f"按优先级分布: 高({summary['by_priority']['high']}) | 中({summary['by_priority']['medium']}) | 低({summary['by_priority']['low']})")
    print(f"按类别分布:")
    for category, count in summary['by_category'].items():
        print(f"  {category}: {count}")
    
    print(f"\n预估测试时间:")
    print(f"  高优先级: {summary['estimated_time']['high_priority']} 分钟")
    print(f"  中优先级: {summary['estimated_time']['medium_priority']} 分钟")
    print(f"  低优先级: {summary['estimated_time']['low_priority']} 分钟")
    print(f"  总计: {summary['estimated_time']['total_hours']:.1f} 小时")
    
    # 保存测试矩阵
    matrix.save_test_matrix('chan_test_matrix.xlsx')
    
    # 显示部分测试场景示例
    scenarios = matrix.generate_all_test_scenarios()
    print(f"\n前5个测试场景示例:")
    for i, scenario in enumerate(scenarios[:5]):
        print(f"{i+1}. {scenario.name}")
        print(f"   描述: {scenario.description}")
        print(f"   优先级: {scenario.priority}")
        print(f"   预期行为: {scenario.expected_behavior}")
        print()

if __name__ == "__main__":
    main()