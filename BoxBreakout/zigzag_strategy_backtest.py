#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZigZag箱体突破策略回测系统
使用OKX标准参数进行策略优化和盈利效果测试

核心功能：
1. 使用OKX标准ZigZag参数（deviation=5.0, depth=10）
2. 测试用户推荐参数（deviation=1.0）的效果
3. 计算策略盈利指标：胜率、盈亏比、最大回撤等
4. 对比不同参数设置下的策略表现

作者: AI Assistant
版本: V1.0
日期: 2025-01-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from typing import List, Dict, Tuple, Optional
import json
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eth_advanced_analyzer import AdvancedBoxBreakoutAnalyzer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ZigZagStrategyBacktest:
    """ZigZag箱体突破策略回测系统"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []  # 持仓记录
        self.trades = []     # 交易记录
        self.equity_curve = []  # 资金曲线
        
    def zigzag_algorithm_optimized(self, highs: np.array, lows: np.array, 
                                  deviation: float = 5.0, depth: int = 10):
        """
        优化的ZigZag算法，支持不同参数配置
        """
        swing_points = []
        zigzag_line = []
        
        if len(highs) < depth or len(lows) < depth:
            return swing_points, zigzag_line
        
        # 寻找局部极值点
        high_indices = argrelextrema(highs, np.greater, order=depth//2)[0]
        low_indices = argrelextrema(lows, np.less, order=depth//2)[0]
        
        # 合并并排序极值点
        all_extrema = []
        for idx in high_indices:
            all_extrema.append((idx, highs[idx], 'high'))
        for idx in low_indices:
            all_extrema.append((idx, lows[idx], 'low'))
        
        all_extrema.sort(key=lambda x: x[0])
        
        if not all_extrema:
            return swing_points, zigzag_line
        
        # 过滤掉幅度不够的摆动
        filtered_points = [all_extrema[0]]
        
        for current in all_extrema[1:]:
            last = filtered_points[-1]
            
            # 计算价格变化幅度
            price_change_pct = abs(current[1] - last[1]) / last[1] * 100
            
            if price_change_pct >= deviation:
                # 如果类型相同，保留更极端的点
                if current[2] == last[2]:
                    if (current[2] == 'high' and current[1] > last[1]) or \
                       (current[2] == 'low' and current[1] < last[1]):
                        filtered_points[-1] = current
                else:
                    filtered_points.append(current)
        
        # 转换为摆动点格式
        for point in filtered_points:
            swing_points.append({
                'index': point[0],
                'price': point[1],
                'type': point[2]
            })
            zigzag_line.append((point[0], point[1]))
        
        return swing_points, zigzag_line
    
    def identify_support_resistance(self, swing_points: List[Dict], 
                                   lookback_period: int = 20) -> List[Dict]:
        """
        基于摆动点识别支撑阻力位
        """
        levels = []
        
        if len(swing_points) < 2:
            return levels
        
        # 提取高点和低点
        highs = [p for p in swing_points if p['type'] == 'high']
        lows = [p for p in swing_points if p['type'] == 'low']
        
        # 识别阻力位（高点聚集区域）
        for i, high in enumerate(highs):
            nearby_highs = []
            for j, other_high in enumerate(highs):
                if i != j and abs(high['index'] - other_high['index']) <= lookback_period:
                    price_diff = abs(high['price'] - other_high['price']) / high['price']
                    if price_diff <= 0.02:  # 2%的价格容差
                        nearby_highs.append(other_high)
            
            if len(nearby_highs) >= 1:  # 至少有一个相近的高点
                avg_price = np.mean([high['price']] + [h['price'] for h in nearby_highs])
                levels.append({
                    'type': 'resistance',
                    'price': avg_price,
                    'strength': len(nearby_highs) + 1,
                    'index': high['index']
                })
        
        # 识别支撑位（低点聚集区域）
        for i, low in enumerate(lows):
            nearby_lows = []
            for j, other_low in enumerate(lows):
                if i != j and abs(low['index'] - other_low['index']) <= lookback_period:
                    price_diff = abs(low['price'] - other_low['price']) / low['price']
                    if price_diff <= 0.02:  # 2%的价格容差
                        nearby_lows.append(other_low)
            
            if len(nearby_lows) >= 1:  # 至少有一个相近的低点
                avg_price = np.mean([low['price']] + [l['price'] for l in nearby_lows])
                levels.append({
                    'type': 'support',
                    'price': avg_price,
                    'strength': len(nearby_lows) + 1,
                    'index': low['index']
                })
        
        # 按强度排序
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return levels
    
    def generate_trading_signals(self, prices: np.array, levels: List[Dict], 
                               volumes: np.array = None) -> List[Dict]:
        """
        基于支撑阻力位生成交易信号
        """
        signals = []
        
        for i in range(1, len(prices)):
            current_price = prices[i]
            prev_price = prices[i-1]
            
            for level in levels:
                level_price = level['price']
                
                # 向上突破阻力位
                if (level['type'] == 'resistance' and 
                    prev_price <= level_price and current_price > level_price):
                    
                    breakout_pct = (current_price - level_price) / level_price * 100
                    
                    # 成交量确认（如果有成交量数据）
                    volume_confirmed = True
                    if volumes is not None and i >= 5:
                        avg_volume = np.mean(volumes[i-5:i])
                        volume_confirmed = volumes[i] > avg_volume * 1.2
                    
                    signal = {
                        'type': 'BUY',
                        'index': i,
                        'price': current_price,
                        'level_price': level_price,
                        'breakout_pct': breakout_pct,
                        'level_strength': level['strength'],
                        'volume_confirmed': volume_confirmed,
                        'stop_loss': level_price * 0.98,  # 2%止损
                        'take_profit': current_price * 1.06  # 6%止盈
                    }
                    signals.append(signal)
                
                # 向下突破支撑位
                elif (level['type'] == 'support' and 
                      prev_price >= level_price and current_price < level_price):
                    
                    breakout_pct = (level_price - current_price) / level_price * 100
                    
                    # 成交量确认
                    volume_confirmed = True
                    if volumes is not None and i >= 5:
                        avg_volume = np.mean(volumes[i-5:i])
                        volume_confirmed = volumes[i] > avg_volume * 1.2
                    
                    signal = {
                        'type': 'SELL',
                        'index': i,
                        'price': current_price,
                        'level_price': level_price,
                        'breakout_pct': breakout_pct,
                        'level_strength': level['strength'],
                        'volume_confirmed': volume_confirmed,
                        'stop_loss': level_price * 1.02,  # 2%止损
                        'take_profit': current_price * 0.94  # 6%止盈
                    }
                    signals.append(signal)
        
        return signals
    
    def execute_backtest(self, prices: np.array, signals: List[Dict], 
                        dates: List = None) -> Dict:
        """
        执行回测
        """
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
        position = None  # 当前持仓
        
        for i, price in enumerate(prices):
            # 检查是否有新信号
            current_signals = [s for s in signals if s['index'] == i]
            
            # 处理平仓
            if position is not None:
                # 检查止损止盈
                if position['type'] == 'LONG':
                    if price <= position['stop_loss'] or price >= position['take_profit']:
                        # 平多仓
                        pnl = (price - position['entry_price']) * position['quantity']
                        self.current_capital += pnl
                        
                        trade = {
                            'entry_date': position['entry_date'],
                            'exit_date': dates[i] if dates else i,
                            'entry_price': position['entry_price'],
                            'exit_price': price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'return_pct': pnl / (position['entry_price'] * position['quantity']) * 100,
                            'type': 'LONG',
                            'exit_reason': 'STOP_LOSS' if price <= position['stop_loss'] else 'TAKE_PROFIT'
                        }
                        self.trades.append(trade)
                        position = None
                
                elif position['type'] == 'SHORT':
                    if price >= position['stop_loss'] or price <= position['take_profit']:
                        # 平空仓
                        pnl = (position['entry_price'] - price) * position['quantity']
                        self.current_capital += pnl
                        
                        trade = {
                            'entry_date': position['entry_date'],
                            'exit_date': dates[i] if dates else i,
                            'entry_price': position['entry_price'],
                            'exit_price': price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'return_pct': pnl / (position['entry_price'] * position['quantity']) * 100,
                            'type': 'SHORT',
                            'exit_reason': 'STOP_LOSS' if price >= position['stop_loss'] else 'TAKE_PROFIT'
                        }
                        self.trades.append(trade)
                        position = None
            
            # 处理开仓
            if position is None and current_signals:
                signal = current_signals[0]  # 取第一个信号
                
                # 只交易强度较高的信号
                if signal['level_strength'] >= 2 and signal['volume_confirmed']:
                    position_size = self.current_capital * 0.1  # 10%仓位
                    quantity = position_size / price
                    
                    if signal['type'] == 'BUY':
                        position = {
                            'type': 'LONG',
                            'entry_date': dates[i] if dates else i,
                            'entry_price': price,
                            'quantity': quantity,
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit']
                        }
                    elif signal['type'] == 'SELL':
                        position = {
                            'type': 'SHORT',
                            'entry_date': dates[i] if dates else i,
                            'entry_price': price,
                            'quantity': quantity,
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit']
                        }
            
            # 记录资金曲线
            current_equity = self.current_capital
            if position is not None:
                if position['type'] == 'LONG':
                    unrealized_pnl = (price - position['entry_price']) * position['quantity']
                else:
                    unrealized_pnl = (position['entry_price'] - price) * position['quantity']
                current_equity += unrealized_pnl
            
            self.equity_curve.append(current_equity)
        
        # 计算回测指标
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        """
        计算策略表现指标
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # 基础指标
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # 最大回撤
        peak = self.initial_capital
        max_drawdown = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 盈亏比
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 夏普比率（简化计算）
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1] * 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.current_capital
        }
    
    def run_parameter_comparison(self, data: pd.DataFrame, 
                               param_configs: List[Dict]) -> Dict:
        """
        运行不同参数配置的对比测试
        """
        results = {}
        
        prices = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values if 'volume' in data.columns else None
        dates = pd.to_datetime(data['date']).tolist()
        
        for config in param_configs:
            print(f"\n测试参数配置: {config['name']}")
            print(f"deviation={config['deviation']}, depth={config['depth']}")
            
            # 重置资金
            self.current_capital = self.initial_capital
            
            # 使用当前参数运行ZigZag算法
            swing_points, zigzag_line = self.zigzag_algorithm_optimized(
                highs, lows, config['deviation'], config['depth']
            )
            
            print(f"识别摆动点: {len(swing_points)} 个")
            
            # 识别支撑阻力位
            levels = self.identify_support_resistance(swing_points)
            print(f"识别支撑阻力位: {len(levels)} 个")
            
            # 生成交易信号
            signals = self.generate_trading_signals(prices, levels, volumes)
            print(f"生成交易信号: {len(signals)} 个")
            
            # 执行回测
            performance = self.execute_backtest(prices, signals, dates)
            
            results[config['name']] = {
                'config': config,
                'swing_points': len(swing_points),
                'levels': len(levels),
                'signals': len(signals),
                'performance': performance,
                'trades': self.trades.copy(),
                'equity_curve': self.equity_curve.copy()
            }
            
            print(f"回测结果:")
            print(f"  总交易次数: {performance['total_trades']}")
            print(f"  胜率: {performance['win_rate']:.2f}%")
            print(f"  总收益率: {performance['total_return']:.2f}%")
            print(f"  最大回撤: {performance['max_drawdown']:.2f}%")
            print(f"  盈亏比: {performance['profit_factor']:.2f}")
        
        return results
    
    def plot_comparison_results(self, results: Dict, save_path: str = None):
        """
        绘制对比结果图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ZigZag策略参数对比分析', fontsize=16, fontweight='bold')
        
        configs = list(results.keys())
        
        # 1. 胜率对比
        win_rates = [results[config]['performance']['win_rate'] for config in configs]
        axes[0, 0].bar(configs, win_rates, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[0, 0].set_title('胜率对比 (%)')
        axes[0, 0].set_ylabel('胜率 (%)')
        for i, v in enumerate(win_rates):
            axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 2. 总收益率对比
        total_returns = [results[config]['performance']['total_return'] for config in configs]
        colors = ['green' if x > 0 else 'red' for x in total_returns]
        axes[0, 1].bar(configs, total_returns, color=colors)
        axes[0, 1].set_title('总收益率对比 (%)')
        axes[0, 1].set_ylabel('收益率 (%)')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        for i, v in enumerate(total_returns):
            axes[0, 1].text(i, v + (1 if v > 0 else -1), f'{v:.1f}%', ha='center', 
                           va='bottom' if v > 0 else 'top')
        
        # 3. 最大回撤对比
        max_drawdowns = [results[config]['performance']['max_drawdown'] for config in configs]
        axes[1, 0].bar(configs, max_drawdowns, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 0].set_title('最大回撤对比 (%)')
        axes[1, 0].set_ylabel('最大回撤 (%)')
        for i, v in enumerate(max_drawdowns):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
        
        # 4. 资金曲线对比
        for config in configs:
            equity_curve = results[config]['equity_curve']
            axes[1, 1].plot(equity_curve, label=config, linewidth=2)
        
        axes[1, 1].set_title('资金曲线对比')
        axes[1, 1].set_ylabel('资金 (元)')
        axes[1, 1].set_xlabel('交易日')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图表已保存到: {save_path}")
        
        plt.show()


def fetch_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取股票数据"""
    import baostock as bs
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return None
        
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,open,high,low,close,volume,amount,turn",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
        
    bs.logout()
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    """主函数"""
    print("ZigZag箱体突破策略回测系统")
    print("=" * 50)
    
    # 创建回测实例
    backtest = ZigZagStrategyBacktest(initial_capital=100000)
    
    # 加载股票数据（使用中兴通讯作为测试）
    data = fetch_stock_data("sz.000063", "2023-01-01", "2025-01-27")
    
    if data is None:
        print("数据加载失败！")
        return
    
    print(f"成功加载 {len(data)} 条股票数据")
    
    # 定义测试参数配置
    param_configs = [
        {
            'name': 'OKX标准参数',
            'deviation': 5.0,
            'depth': 10,
            'description': 'OKX官方标准参数'
        },
        {
            'name': '用户推荐参数',
            'deviation': 1.0,
            'depth': 10,
            'description': '用户测试效果良好的参数'
        },
        {
            'name': '原代码库参数',
            'deviation': 3.5,
            'depth': 6,
            'description': '原代码库使用的参数'
        }
    ]
    
    # 运行参数对比测试
    print("\n开始参数对比测试...")
    results = backtest.run_parameter_comparison(data, param_configs)
    
    # 绘制对比结果
    backtest.plot_comparison_results(results, 'zigzag_strategy_comparison.png')
    
    # 保存详细结果
    summary_results = {}
    for config_name, result in results.items():
        summary_results[config_name] = {
            'config': result['config'],
            'performance': result['performance'],
            'signal_stats': {
                'swing_points': result['swing_points'],
                'levels': result['levels'],
                'signals': result['signals']
            }
        }
    
    with open('zigzag_strategy_backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n详细回测结果已保存到: zigzag_strategy_backtest_results.json")
    
    # 输出最佳参数推荐
    best_config = max(results.keys(), 
                     key=lambda x: results[x]['performance']['total_return'])
    
    print(f"\n🏆 最佳参数配置: {best_config}")
    print(f"   总收益率: {results[best_config]['performance']['total_return']:.2f}%")
    print(f"   胜率: {results[best_config]['performance']['win_rate']:.2f}%")
    print(f"   最大回撤: {results[best_config]['performance']['max_drawdown']:.2f}%")


if __name__ == "__main__":
    main()