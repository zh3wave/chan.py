"""
Chan.py 批量参数测试执行器
用于自动化执行大量参数组合测试并收集结果

Version: 1.0
Created: 2024
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Chan import CChan
from ChanConfig import CChanConfig
from KLine.KLine_List import CKLine_List
from chan_test_matrix import ChanTestMatrix, TestScenario
from chan_parameter_tester import ChanParameterTester

class ChanBatchTester:
    """Chan.py 批量参数测试执行器"""
    
    def __init__(self, data_file: str = None, output_dir: str = "batch_test_results"):
        self.data_file = data_file or "eth_data.csv"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "configs").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        self.test_matrix = ChanTestMatrix()
        self.parameter_tester = ChanParameterTester()
        self.test_results = []
        self.failed_tests = []
        
        # 测试统计
        self.stats = {
            'total_tests': 0,
            'completed_tests': 0,
            'failed_tests': 0,
            'start_time': None,
            'end_time': None,
            'duration': 0
        }
    
    def load_test_data(self) -> Optional[pd.DataFrame]:
        """加载测试数据"""
        try:
            if os.path.exists(self.data_file):
                print(f"加载测试数据: {self.data_file}")
                return pd.read_csv(self.data_file)
            else:
                print(f"数据文件不存在: {self.data_file}")
                print("生成模拟数据用于测试...")
                return self._generate_mock_data()
        except Exception as e:
            print(f"加载数据失败: {e}")
            return self._generate_mock_data()
    
    def _generate_mock_data(self, days: int = 500) -> pd.DataFrame:
        """生成模拟数据"""
        print(f"生成 {days} 天的模拟ETH数据...")
        
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # 生成模拟价格数据
        np.random.seed(42)
        base_price = 2000
        price_changes = np.random.normal(0, 0.02, days)  # 2%的日波动
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))  # 最低价格100
        
        # 生成OHLCV数据
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # 生成当日高低开收
            daily_range = close * 0.05  # 5%的日内波动
            high = close + np.random.uniform(0, daily_range)
            low = close - np.random.uniform(0, daily_range)
            open_price = prices[i-1] if i > 0 else close
            
            # 确保价格关系合理
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.uniform(1000000, 10000000)  # 随机成交量
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': int(volume)
            })
        
        df = pd.DataFrame(data)
        
        # 保存模拟数据
        mock_file = self.output_dir / "mock_eth_data.csv"
        df.to_csv(mock_file, index=False)
        print(f"模拟数据已保存到: {mock_file}")
        
        return df
    
    def create_chan_instance(self, config: Dict[str, Any], data: pd.DataFrame) -> Optional[CChan]:
        """创建Chan实例"""
        try:
            # 创建配置
            chan_config = CChanConfig()
            
            # 设置笔配置
            if 'bi_algo' in config:
                chan_config.bi_conf.bi_algo = config['bi_algo']
            if 'bi_strict' in config:
                chan_config.bi_conf.is_strict = config['bi_strict']
            if 'bi_fx_check' in config:
                chan_config.bi_conf.bi_fx_check = config['bi_fx_check']
            if 'gap_as_kl' in config:
                chan_config.bi_conf.gap_as_kl = config['gap_as_kl']
            if 'bi_end_is_peak' in config:
                chan_config.bi_conf.bi_end_is_peak = config['bi_end_is_peak']
            if 'bi_allow_sub_peak' in config:
                chan_config.bi_conf.bi_allow_sub_peak = config['bi_allow_sub_peak']
            
            # 设置线段配置
            if 'seg_algo' in config:
                chan_config.seg_conf.seg_algo = config['seg_algo']
            if 'left_seg_method' in config:
                chan_config.seg_conf.left_method = config['left_seg_method']
            
            # 设置中枢配置
            if 'zs_algo' in config:
                chan_config.zs_conf.zs_algo = config['zs_algo']
            if 'need_combine' in config:
                chan_config.zs_conf.need_combine = config['need_combine']
            if 'zs_combine_mode' in config:
                chan_config.zs_conf.zs_combine_mode = config['zs_combine_mode']
            if 'one_bi_zs' in config:
                chan_config.zs_conf.one_bi_zs = config['one_bi_zs']
            
            # 设置买卖点配置
            bsp_config = {}
            if 'divergence_rate' in config:
                bsp_config['divergence_rate'] = config['divergence_rate']
            if 'min_zs_cnt' in config:
                bsp_config['min_zs_cnt'] = config['min_zs_cnt']
            if 'macd_algo' in config:
                bsp_config['macd_algo'] = config['macd_algo']
            if 'bs_type' in config:
                bsp_config['bs_type'] = config['bs_type']
            
            if bsp_config:
                chan_config.set_bsp_config(**bsp_config)
            
            # 设置技术指标配置
            if 'macd_fast' in config:
                chan_config.macd_conf.fast = config['macd_fast']
            if 'macd_slow' in config:
                chan_config.macd_conf.slow = config['macd_slow']
            if 'macd_signal' in config:
                chan_config.macd_conf.signal = config['macd_signal']
            if 'rsi_period' in config:
                chan_config.rsi_conf.period = config['rsi_period']
            
            # 创建K线数据
            kline_list = CKLine_List()
            for _, row in data.iterrows():
                kline_list.add_kline(
                    time=row['date'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
            
            # 创建Chan实例
            chan = CChan(
                code="ETH_TEST",
                begin_time=data.iloc[0]['date'],
                end_time=data.iloc[-1]['date'],
                data_src="custom",
                lv_list=["1d"],
                config=chan_config,
                autype=None
            )
            
            # 手动设置K线数据
            chan[0] = kline_list
            
            return chan
            
        except Exception as e:
            print(f"创建Chan实例失败: {e}")
            traceback.print_exc()
            return None
    
    def extract_metrics(self, chan: CChan, config: Dict[str, Any]) -> Dict[str, Any]:
        """提取分析指标"""
        try:
            metrics = {
                'config': config.copy(),
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'error': None
            }
            
            # 基础统计
            kline_count = len(chan[0]) if chan[0] else 0
            bi_count = len(chan[0].bi_list) if chan[0] and hasattr(chan[0], 'bi_list') else 0
            seg_count = len(chan[0].seg_list) if chan[0] and hasattr(chan[0], 'seg_list') else 0
            zs_count = len(chan[0].zs_list) if chan[0] and hasattr(chan[0], 'zs_list') else 0
            
            metrics.update({
                'kline_count': kline_count,
                'bi_count': bi_count,
                'seg_count': seg_count,
                'zs_count': zs_count,
                'bi_ratio': bi_count / kline_count if kline_count > 0 else 0,
                'seg_ratio': seg_count / kline_count if kline_count > 0 else 0,
                'zs_ratio': zs_count / kline_count if kline_count > 0 else 0,
            })
            
            # 买卖点统计
            if hasattr(chan[0], 'bs_point_lst') and chan[0].bs_point_lst:
                bs_points = chan[0].bs_point_lst
                buy_points = [p for p in bs_points if p.is_buy]
                sell_points = [p for p in bs_points if not p.is_buy]
                
                metrics.update({
                    'total_bs_points': len(bs_points),
                    'buy_points': len(buy_points),
                    'sell_points': len(sell_points),
                    'bs_point_ratio': len(bs_points) / kline_count if kline_count > 0 else 0,
                })
            else:
                metrics.update({
                    'total_bs_points': 0,
                    'buy_points': 0,
                    'sell_points': 0,
                    'bs_point_ratio': 0,
                })
            
            # 计算模拟收益率（简化版本）
            if kline_count > 0:
                first_price = chan[0][0].close
                last_price = chan[0][-1].close
                total_return = (last_price - first_price) / first_price
                metrics['total_return'] = total_return
            else:
                metrics['total_return'] = 0
            
            # 复杂度指标
            metrics.update({
                'complexity_score': bi_count + seg_count * 2 + zs_count * 3,
                'efficiency_score': (seg_count + zs_count) / bi_count if bi_count > 0 else 0,
            })
            
            return metrics
            
        except Exception as e:
            return {
                'config': config.copy(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'kline_count': 0,
                'bi_count': 0,
                'seg_count': 0,
                'zs_count': 0,
            }
    
    def run_single_test(self, scenario: TestScenario, data: pd.DataFrame) -> Dict[str, Any]:
        """运行单个测试"""
        print(f"运行测试: {scenario.name}")
        
        start_time = time.time()
        
        try:
            # 处理对比测试
            if 'config1' in scenario.config:
                # 对比测试
                config1 = scenario.config['config1'].copy()
                config2 = scenario.config['config2'].copy()
                
                # 移除name字段
                config1.pop('name', None)
                config2.pop('name', None)
                
                chan1 = self.create_chan_instance(config1, data)
                chan2 = self.create_chan_instance(config2, data)
                
                if chan1 and chan2:
                    metrics1 = self.extract_metrics(chan1, config1)
                    metrics2 = self.extract_metrics(chan2, config2)
                    
                    result = {
                        'test_name': scenario.name,
                        'test_type': 'comparison',
                        'description': scenario.description,
                        'priority': scenario.priority,
                        'expected_behavior': scenario.expected_behavior,
                        'config1': config1,
                        'config2': config2,
                        'metrics1': metrics1,
                        'metrics2': metrics2,
                        'execution_time': time.time() - start_time,
                        'success': True,
                        'error': None
                    }
                else:
                    result = {
                        'test_name': scenario.name,
                        'test_type': 'comparison',
                        'success': False,
                        'error': 'Failed to create Chan instances',
                        'execution_time': time.time() - start_time
                    }
            else:
                # 单配置测试
                chan = self.create_chan_instance(scenario.config, data)
                
                if chan:
                    metrics = self.extract_metrics(chan, scenario.config)
                    
                    result = {
                        'test_name': scenario.name,
                        'test_type': 'single',
                        'description': scenario.description,
                        'priority': scenario.priority,
                        'expected_behavior': scenario.expected_behavior,
                        'config': scenario.config,
                        'metrics': metrics,
                        'execution_time': time.time() - start_time,
                        'success': True,
                        'error': None
                    }
                else:
                    result = {
                        'test_name': scenario.name,
                        'test_type': 'single',
                        'success': False,
                        'error': 'Failed to create Chan instance',
                        'execution_time': time.time() - start_time
                    }
            
            return result
            
        except Exception as e:
            return {
                'test_name': scenario.name,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'traceback': traceback.format_exc()
            }
    
    def run_batch_tests(self, 
                       priority_filter: List[str] = None,
                       category_filter: List[str] = None,
                       max_workers: int = 4,
                       max_tests: int = None) -> Dict[str, Any]:
        """运行批量测试"""
        
        print("=== 开始批量参数测试 ===\n")
        
        # 加载测试数据
        data = self.load_test_data()
        if data is None:
            print("无法加载测试数据，测试终止")
            return {'success': False, 'error': 'No test data available'}
        
        # 生成测试场景
        all_scenarios = self.test_matrix.generate_all_test_scenarios()
        
        # 应用过滤器
        filtered_scenarios = all_scenarios
        
        if priority_filter:
            filtered_scenarios = [s for s in filtered_scenarios if s.priority in priority_filter]
        
        if category_filter:
            filtered_scenarios = [s for s in filtered_scenarios 
                                if any(s.name.startswith(cat) for cat in category_filter)]
        
        if max_tests:
            filtered_scenarios = filtered_scenarios[:max_tests]
        
        print(f"总测试场景: {len(all_scenarios)}")
        print(f"过滤后场景: {len(filtered_scenarios)}")
        
        # 初始化统计
        self.stats['total_tests'] = len(filtered_scenarios)
        self.stats['start_time'] = datetime.now()
        
        # 执行测试
        if max_workers == 1:
            # 单线程执行
            for i, scenario in enumerate(filtered_scenarios):
                print(f"\n进度: {i+1}/{len(filtered_scenarios)}")
                result = self.run_single_test(scenario, data)
                
                if result['success']:
                    self.test_results.append(result)
                    self.stats['completed_tests'] += 1
                else:
                    self.failed_tests.append(result)
                    self.stats['failed_tests'] += 1
                    print(f"测试失败: {result.get('error', 'Unknown error')}")
        else:
            # 多线程执行
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_scenario = {
                    executor.submit(self.run_single_test, scenario, data): scenario 
                    for scenario in filtered_scenarios
                }
                
                for i, future in enumerate(as_completed(future_to_scenario)):
                    print(f"\n进度: {i+1}/{len(filtered_scenarios)}")
                    
                    try:
                        result = future.result()
                        
                        if result['success']:
                            self.test_results.append(result)
                            self.stats['completed_tests'] += 1
                        else:
                            self.failed_tests.append(result)
                            self.stats['failed_tests'] += 1
                            print(f"测试失败: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        self.stats['failed_tests'] += 1
                        print(f"测试执行异常: {e}")
        
        # 完成统计
        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print(f"\n=== 批量测试完成 ===")
        print(f"总测试数: {self.stats['total_tests']}")
        print(f"成功测试: {self.stats['completed_tests']}")
        print(f"失败测试: {self.stats['failed_tests']}")
        print(f"总耗时: {self.stats['duration']:.1f} 秒")
        
        # 保存结果
        self.save_results()
        
        return {
            'success': True,
            'stats': self.stats,
            'results_count': len(self.test_results),
            'failed_count': len(self.failed_tests)
        }
    
    def save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = self.output_dir / "results" / f"batch_test_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.stats,
                'test_results': self.test_results,
                'failed_tests': self.failed_tests
            }, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"详细结果已保存到: {results_file}")
        
        # 保存汇总表格
        if self.test_results:
            summary_data = []
            
            for result in self.test_results:
                if result['test_type'] == 'single':
                    metrics = result.get('metrics', {})
                    row = {
                        'test_name': result['test_name'],
                        'priority': result['priority'],
                        'execution_time': result['execution_time'],
                        'kline_count': metrics.get('kline_count', 0),
                        'bi_count': metrics.get('bi_count', 0),
                        'seg_count': metrics.get('seg_count', 0),
                        'zs_count': metrics.get('zs_count', 0),
                        'total_bs_points': metrics.get('total_bs_points', 0),
                        'total_return': metrics.get('total_return', 0),
                        'complexity_score': metrics.get('complexity_score', 0),
                        'efficiency_score': metrics.get('efficiency_score', 0),
                    }
                    
                    # 添加配置参数
                    config = result.get('config', {})
                    for key, value in config.items():
                        row[f'config_{key}'] = value
                    
                    summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_file = self.output_dir / "results" / f"batch_test_summary_{timestamp}.xlsx"
                summary_df.to_excel(summary_file, index=False)
                print(f"汇总表格已保存到: {summary_file}")
        
        # 生成测试报告
        self.generate_test_report(timestamp)
    
    def generate_test_report(self, timestamp: str):
        """生成测试报告"""
        report_file = self.output_dir / "reports" / f"test_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Chan.py 批量参数测试报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 测试统计
            f.write("## 测试统计\n\n")
            f.write(f"- 总测试数: {self.stats['total_tests']}\n")
            f.write(f"- 成功测试: {self.stats['completed_tests']}\n")
            f.write(f"- 失败测试: {self.stats['failed_tests']}\n")
            f.write(f"- 成功率: {self.stats['completed_tests']/self.stats['total_tests']*100:.1f}%\n")
            f.write(f"- 总耗时: {self.stats['duration']:.1f} 秒\n")
            f.write(f"- 平均耗时: {self.stats['duration']/self.stats['total_tests']:.2f} 秒/测试\n\n")
            
            # 失败测试分析
            if self.failed_tests:
                f.write("## 失败测试分析\n\n")
                for failed in self.failed_tests[:10]:  # 只显示前10个
                    f.write(f"- **{failed['test_name']}**: {failed.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            # 成功测试摘要
            if self.test_results:
                f.write("## 成功测试摘要\n\n")
                
                # 按优先级统计
                priority_stats = {}
                for result in self.test_results:
                    priority = result.get('priority', 'unknown')
                    priority_stats[priority] = priority_stats.get(priority, 0) + 1
                
                f.write("### 按优先级分布\n\n")
                for priority, count in priority_stats.items():
                    f.write(f"- {priority}: {count}\n")
                f.write("\n")
                
                # 性能指标统计
                if any(r['test_type'] == 'single' for r in self.test_results):
                    single_results = [r for r in self.test_results if r['test_type'] == 'single']
                    
                    bi_counts = [r['metrics'].get('bi_count', 0) for r in single_results]
                    seg_counts = [r['metrics'].get('seg_count', 0) for r in single_results]
                    zs_counts = [r['metrics'].get('zs_count', 0) for r in single_results]
                    
                    f.write("### 性能指标统计\n\n")
                    f.write(f"- 笔数量: 平均 {np.mean(bi_counts):.1f}, 范围 {min(bi_counts)}-{max(bi_counts)}\n")
                    f.write(f"- 线段数量: 平均 {np.mean(seg_counts):.1f}, 范围 {min(seg_counts)}-{max(seg_counts)}\n")
                    f.write(f"- 中枢数量: 平均 {np.mean(zs_counts):.1f}, 范围 {min(zs_counts)}-{max(zs_counts)}\n")
                    f.write("\n")
            
            f.write("## 建议\n\n")
            f.write("1. 重点关注高优先级测试的结果差异\n")
            f.write("2. 分析失败测试的共同原因\n")
            f.write("3. 对比不同参数组合的性能指标\n")
            f.write("4. 选择最适合当前数据特征的参数配置\n")
        
        print(f"测试报告已保存到: {report_file}")

def main():
    """主函数 - 演示批量测试"""
    print("=== Chan.py 批量参数测试器 ===\n")
    
    # 创建批量测试器
    tester = ChanBatchTester()
    
    # 运行高优先级测试（限制数量以节省时间）
    result = tester.run_batch_tests(
        priority_filter=['high'],
        max_tests=20,  # 限制测试数量
        max_workers=2  # 使用2个线程
    )
    
    if result['success']:
        print(f"\n批量测试完成!")
        print(f"成功测试: {result['results_count']}")
        print(f"失败测试: {result['failed_count']}")
        print(f"结果已保存到: batch_test_results/")
    else:
        print(f"批量测试失败: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()