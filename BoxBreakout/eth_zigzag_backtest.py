#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH ZigZag策略回测系统
专门针对ETH数据进行ZigZag箱体突破策略回测
使用OKX标准参数进行优化测试

功能：
1. 加载ETH 5分钟K线数据
2. 使用不同ZigZag参数进行回测
3. 计算策略盈利指标
4. 对比参数效果
5. 生成可视化报告

作者: AI Assistant
版本: V1.0
日期: 2025年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_zigzag_standard import OKXZigZag

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ETHZigZagBacktester:
    """ETH ZigZag策略回测器"""
    
    def __init__(self):
        self.data = None
        self.results = {}
        
    def load_eth_data(self, file_path: str, start_index: int = 200000, length: int = 5000) -> bool:
        """加载ETH数据"""
        try:
            # 读取完整数据
            full_data = pd.read_csv(file_path)
            print(f"✅ 成功加载ETH数据，总共 {len(full_data)} 条记录")
            
            # 选择指定段的数据进行回测
            end_index = min(start_index + length, len(full_data))
            self.data = full_data.iloc[start_index:end_index].copy()
            
            # 转换日期格式
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            print(f"📊 回测数据段: 第 {start_index} 到 {end_index-1} 条记录")
            print(f"📅 时间范围: {self.data['date'].iloc[0]} 到 {self.data['date'].iloc[-1]}")
            print(f"💰 价格范围: ${self.data['low'].min():.2f} - ${self.data['high'].max():.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载ETH数据失败: {e}")
            return False
    
    def calculate_macd(self, prices: np.array, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> dict:
        """计算MACD指标"""
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_values = np.zeros_like(data)
            ema_values[0] = data[0]
            for i in range(1, len(data)):
                ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
            return ema_values
        
        fast_ema = ema(prices, fast_period)
        slow_ema = ema(prices, slow_period)
        dif = fast_ema - slow_ema
        dea = ema(dif, signal_period)
        macd = (dif - dea) * 2
        
        return {'dif': dif, 'dea': dea, 'macd': macd}
    
    def identify_support_resistance(self, swing_points: List[Dict], price_tolerance: float = 0.02) -> List[Dict]:
        """识别支撑阻力位"""
        if len(swing_points) < 2:
            return []
        
        levels = []
        
        # 分离高点和低点
        highs = [p for p in swing_points if p['type'] == 'high']
        lows = [p for p in swing_points if p['type'] == 'low']
        
        # 识别阻力位（高点聚集）
        for i, high1 in enumerate(highs):
            cluster = [high1]
            for j, high2 in enumerate(highs):
                if i != j:
                    price_diff = abs(high1['price'] - high2['price']) / high1['price']
                    if price_diff <= price_tolerance:
                        cluster.append(high2)
            
            if len(cluster) >= 2:  # 至少2个点形成阻力位
                avg_price = np.mean([p['price'] for p in cluster])
                levels.append({
                    'type': 'resistance',
                    'price': avg_price,
                    'strength': len(cluster),
                    'points': cluster
                })
        
        # 识别支撑位（低点聚集）
        for i, low1 in enumerate(lows):
            cluster = [low1]
            for j, low2 in enumerate(lows):
                if i != j:
                    price_diff = abs(low1['price'] - low2['price']) / low1['price']
                    if price_diff <= price_tolerance:
                        cluster.append(low2)
            
            if len(cluster) >= 2:  # 至少2个点形成支撑位
                avg_price = np.mean([p['price'] for p in cluster])
                levels.append({
                    'type': 'support',
                    'price': avg_price,
                    'strength': len(cluster),
                    'points': cluster
                })
        
        # 去重并排序
        unique_levels = []
        for level in levels:
            is_duplicate = False
            for existing in unique_levels:
                if (level['type'] == existing['type'] and 
                    abs(level['price'] - existing['price']) / level['price'] < price_tolerance):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_levels.append(level)
        
        return sorted(unique_levels, key=lambda x: x['price'])
    
    def detect_breakout_signals(self, levels: List[Dict], macd: dict, volume_ratio: np.array) -> List[Dict]:
        """检测突破信号"""
        signals = []
        
        if not levels:
            return signals
        
        prices = self.data['close'].values
        volumes = self.data['volume'].values
        
        for i in range(50, len(prices)):  # 从第50根K线开始检测
            current_price = prices[i]
            current_volume = volumes[i]
            
            # 检查每个支撑阻力位
            for level in levels:
                level_price = level['price']
                level_type = level['type']
                
                # 向上突破阻力位
                if (level_type == 'resistance' and 
                    current_price > level_price * 1.005 and  # 突破0.5%
                    prices[i-1] <= level_price):
                    
                    # MACD确认
                    macd_bullish = macd['dif'][i] > macd['dea'][i] and macd['macd'][i] > 0
                    
                    # 成交量确认
                    volume_confirm = volume_ratio[i] > 1.2
                    
                    signal_strength = 0.5
                    if macd_bullish:
                        signal_strength += 0.3
                    if volume_confirm:
                        signal_strength += 0.2
                    
                    signals.append({
                        'index': i,
                        'type': 'buy',
                        'price': current_price,
                        'level_price': level_price,
                        'level_type': level_type,
                        'strength': signal_strength,
                        'macd_confirm': macd_bullish,
                        'volume_confirm': volume_confirm,
                        'stop_loss': level_price * 0.98,  # 2%止损
                        'take_profit': current_price * 1.06  # 6%止盈
                    })
                
                # 向下突破支撑位
                elif (level_type == 'support' and 
                      current_price < level_price * 0.995 and  # 突破0.5%
                      prices[i-1] >= level_price):
                    
                    # MACD确认
                    macd_bearish = macd['dif'][i] < macd['dea'][i] and macd['macd'][i] < 0
                    
                    # 成交量确认
                    volume_confirm = volume_ratio[i] > 1.2
                    
                    signal_strength = 0.5
                    if macd_bearish:
                        signal_strength += 0.3
                    if volume_confirm:
                        signal_strength += 0.2
                    
                    signals.append({
                        'index': i,
                        'type': 'sell',
                        'price': current_price,
                        'level_price': level_price,
                        'level_type': level_type,
                        'strength': signal_strength,
                        'macd_confirm': macd_bearish,
                        'volume_confirm': volume_confirm,
                        'stop_loss': level_price * 1.02,  # 2%止损
                        'take_profit': current_price * 0.94  # 6%止盈
                    })
        
        return signals
    
    def execute_backtest(self, signals: List[Dict]) -> Dict:
        """执行回测"""
        if not signals:
            return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'profit_loss_ratio': 0,
            'trades': [],
            'final_portfolio_value': 10000
        }
        
        prices = self.data['close'].values
        trades = []
        portfolio_value = 10000  # 初始资金
        peak_value = portfolio_value
        max_drawdown = 0
        
        for signal in signals:
            entry_index = signal['index']
            entry_price = signal['price']
            signal_type = signal['type']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            
            # 寻找出场点
            exit_index = None
            exit_price = None
            exit_reason = None
            
            for i in range(entry_index + 1, min(entry_index + 100, len(prices))):  # 最多持仓100根K线
                current_price = prices[i]
                
                if signal_type == 'buy':
                    # 多头止盈止损
                    if current_price >= take_profit:
                        exit_index = i
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                        break
                    elif current_price <= stop_loss:
                        exit_index = i
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                        break
                else:  # sell
                    # 空头止盈止损
                    if current_price <= take_profit:
                        exit_index = i
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                        break
                    elif current_price >= stop_loss:
                        exit_index = i
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                        break
            
            # 如果没有触发止盈止损，按最后价格平仓
            if exit_index is None:
                exit_index = min(entry_index + 100, len(prices) - 1)
                exit_price = prices[exit_index]
                exit_reason = 'time_exit'
            
            # 计算收益
            if signal_type == 'buy':
                return_pct = (exit_price - entry_price) / entry_price
            else:  # sell (做空)
                return_pct = (entry_price - exit_price) / entry_price
            
            # 更新组合价值
            portfolio_value *= (1 + return_pct)
            
            # 计算最大回撤
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            else:
                drawdown = (peak_value - portfolio_value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)
            
            trades.append({
                'entry_index': entry_index,
                'exit_index': exit_index,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'signal_type': signal_type,
                'return_pct': return_pct,
                'exit_reason': exit_reason,
                'strength': signal['strength']
            })
        
        # 计算统计指标
        if trades:
            winning_trades = [t for t in trades if t['return_pct'] > 0]
            losing_trades = [t for t in trades if t['return_pct'] <= 0]
            
            win_rate = len(winning_trades) / len(trades)
            total_return = (portfolio_value - 10000) / 10000
            
            if winning_trades and losing_trades:
                avg_win = np.mean([t['return_pct'] for t in winning_trades])
                avg_loss = abs(np.mean([t['return_pct'] for t in losing_trades]))
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            else:
                profit_loss_ratio = 0
        else:
            win_rate = 0
            total_return = 0
            profit_loss_ratio = 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'profit_loss_ratio': profit_loss_ratio,
            'trades': trades,
            'final_portfolio_value': portfolio_value
        }
    
    def run_parameter_comparison(self, test_configs: List[Dict]) -> Dict:
        """运行参数对比测试"""
        results = {}
        
        prices = self.data['close'].values
        highs = self.data['high'].values
        lows = self.data['low'].values
        volumes = self.data['volume'].values
        
        # 计算技术指标
        macd = self.calculate_macd(prices)
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        for config in test_configs:
            config_name = config['name']
            deviation = config['deviation']
            depth = config['depth']
            
            print(f"\n🔄 测试配置: {config_name}")
            print(f"   参数: deviation={deviation}%, depth={depth}")
            
            # 使用OKX ZigZag算法
            zigzag = OKXZigZag(deviation=deviation, depth=depth)
            swing_points, zigzag_line = zigzag.calculate(highs, lows)
            
            print(f"   识别摆动点: {len(swing_points)} 个")
            
            # 识别支撑阻力位
            levels = self.identify_support_resistance(swing_points)
            print(f"   支撑阻力位: {len(levels)} 个")
            
            # 检测交易信号
            signals = self.detect_breakout_signals(levels, macd, volume_ratio)
            print(f"   交易信号: {len(signals)} 个")
            
            # 执行回测
            backtest_result = self.execute_backtest(signals)
            
            # 保存结果
            results[config_name] = {
                'parameters': {'deviation': deviation, 'depth': depth},
                'swing_points_count': len(swing_points),
                'levels_count': len(levels),
                'signals_count': len(signals),
                'backtest': backtest_result,
                'swing_points': swing_points,
                'levels': levels,
                'signals': signals
            }
            
            print(f"   回测结果: 交易{backtest_result['total_trades']}次, "
                  f"胜率{backtest_result['win_rate']:.1%}, "
                  f"总收益{backtest_result['total_return']:.2%}")
        
        return results
    
    def calculate_volume_ratio(self, volumes: np.array, period: int = 5) -> np.array:
        """计算量比指标"""
        volume_ratio = np.ones_like(volumes, dtype=float)
        
        for i in range(period, len(volumes)):
            avg_volume = np.mean(volumes[i-period:i])
            if avg_volume > 0:
                volume_ratio[i] = volumes[i] / avg_volume
        
        return volume_ratio
    
    def plot_comparison_results(self, results: Dict, save_path: str = None):
        """绘制对比结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ETH ZigZag策略参数对比分析', fontsize=16, fontweight='bold')
        
        config_names = list(results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 1. 摆动点数量对比
        ax1 = axes[0, 0]
        swing_counts = [results[name]['swing_points_count'] for name in config_names]
        bars1 = ax1.bar(config_names, swing_counts, color=colors[:len(config_names)])
        ax1.set_title('摆动点数量对比')
        ax1.set_ylabel('摆动点数量')
        for i, v in enumerate(swing_counts):
            ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # 2. 交易信号数量对比
        ax2 = axes[0, 1]
        signal_counts = [results[name]['signals_count'] for name in config_names]
        bars2 = ax2.bar(config_names, signal_counts, color=colors[:len(config_names)])
        ax2.set_title('交易信号数量对比')
        ax2.set_ylabel('信号数量')
        for i, v in enumerate(signal_counts):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # 3. 胜率对比
        ax3 = axes[1, 0]
        win_rates = [results[name]['backtest']['win_rate'] * 100 for name in config_names]
        bars3 = ax3.bar(config_names, win_rates, color=colors[:len(config_names)])
        ax3.set_title('胜率对比')
        ax3.set_ylabel('胜率 (%)')
        ax3.set_ylim(0, 100)
        for i, v in enumerate(win_rates):
            ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 4. 总收益率对比
        ax4 = axes[1, 1]
        total_returns = [results[name]['backtest']['total_return'] * 100 for name in config_names]
        bars4 = ax4.bar(config_names, total_returns, color=colors[:len(config_names)])
        ax4.set_title('总收益率对比')
        ax4.set_ylabel('总收益率 (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        for i, v in enumerate(total_returns):
            ax4.text(i, v + (1 if v >= 0 else -2), f'{v:.1f}%', ha='center', 
                    va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 对比图表已保存: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, file_path: str):
        """保存回测结果"""
        # 转换为可序列化的格式
        serializable_results = {}
        for config_name, result in results.items():
            serializable_results[config_name] = {
                'parameters': result['parameters'],
                'swing_points_count': result['swing_points_count'],
                'levels_count': result['levels_count'],
                'signals_count': result['signals_count'],
                'backtest': {
                    'total_trades': result['backtest']['total_trades'],
                    'win_rate': result['backtest']['win_rate'],
                    'total_return': result['backtest']['total_return'],
                    'max_drawdown': result['backtest']['max_drawdown'],
                    'profit_loss_ratio': result['backtest']['profit_loss_ratio'],
                    'final_portfolio_value': result['backtest']['final_portfolio_value']
                }
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 回测结果已保存: {file_path}")

def main():
    """主函数"""
    print("🚀 ETH ZigZag策略回测系统启动")
    
    # 初始化回测器
    backtester = ETHZigZagBacktester()
    
    # 加载ETH数据
    eth_file = "../ETH_USDT_5m.csv"
    if not backtester.load_eth_data(eth_file, start_index=200000, length=5000):
        return
    
    # 测试配置
    test_configs = [
        {
            'name': 'OKX标准参数',
            'deviation': 5.0,
            'depth': 10
        },
        {
            'name': '用户推荐参数',
            'deviation': 1.0,
            'depth': 10
        },
        {
            'name': '原代码库参数',
            'deviation': 3.5,
            'depth': 6
        },
        {
            'name': '高敏感度参数',
            'deviation': 0.5,
            'depth': 5
        }
    ]
    
    print(f"\n📋 将测试 {len(test_configs)} 种参数配置:")
    for config in test_configs:
        print(f"   - {config['name']}: deviation={config['deviation']}%, depth={config['depth']}")
    
    # 运行参数对比测试
    results = backtester.run_parameter_comparison(test_configs)
    
    # 生成报告
    print("\n" + "="*60)
    print("📊 ETH ZigZag策略回测结果汇总")
    print("="*60)
    
    for config_name, result in results.items():
        backtest = result['backtest']
        print(f"\n🔍 {config_name}:")
        print(f"   参数: deviation={result['parameters']['deviation']}%, depth={result['parameters']['depth']}")
        print(f"   摆动点: {result['swing_points_count']} 个")
        print(f"   支撑阻力位: {result['levels_count']} 个")
        print(f"   交易信号: {result['signals_count']} 个")
        print(f"   总交易次数: {backtest['total_trades']} 次")
        print(f"   胜率: {backtest['win_rate']:.2%}")
        print(f"   总收益率: {backtest['total_return']:.2%}")
        print(f"   最大回撤: {backtest['max_drawdown']:.2%}")
        print(f"   盈亏比: {backtest['profit_loss_ratio']:.2f}")
        print(f"   最终资金: ${backtest['final_portfolio_value']:.2f}")
    
    # 保存结果和图表
    backtester.save_results(results, 'eth_zigzag_backtest_results.json')
    backtester.plot_comparison_results(results, 'eth_zigzag_comparison.png')
    
    print("\n✅ ETH ZigZag策略回测完成！")

if __name__ == "__main__":
    main()