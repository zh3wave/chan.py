"""
优化版ZigZag分析器 - 集成实战化箱体策略
作者: ZigZag策略团队
版本: 2.0
日期: 2025-01-27

主要改进：
1. 替换原有的简单箱体逻辑为实战化箱体策略
2. 基于真实的支撑阻力位识别
3. 考虑成交量确认和时间因素
4. 提供更准确的交易信号
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_zigzag_standard import OKXZigZag
from BoxBreakout.practical_box_strategy import PracticalBoxStrategy

class OptimizedZigZagAnalyzer:
    """优化版ZigZag分析器"""
    
    def __init__(self, data: pd.DataFrame, deviation: float = 1.0, depth: int = 10):
        """
        初始化分析器
        
        Args:
            data: OHLCV数据
            deviation: ZigZag偏差参数 (%)
            depth: ZigZag深度参数
        """
        self.data = data.copy()
        self.data.reset_index(drop=True, inplace=True)
        self.deviation = deviation
        self.depth = depth
        
        # 初始化ZigZag算法
        self.zigzag = OKXZigZag(deviation=deviation, depth=depth)
        
        # 初始化实战化箱体策略
        self.box_strategy = PracticalBoxStrategy(self.data)
        
        # 计算ZigZag摆动点
        self.swing_points = self._calculate_swing_points()
        
    def _calculate_swing_points(self) -> List[Dict]:
        """计算ZigZag摆动点"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        
        # 使用修复后的ZigZag算法
        swing_points, _ = self.zigzag.calculate(highs, lows)
        
        # 转换为统一格式
        formatted_points = []
        for point in swing_points:
            formatted_points.append({
                'index': point['index'],
                'price': point['price'],
                'type': point['type'],
                'timestamp': self.data.index[point['index']] if hasattr(self.data.index, 'to_pydatetime') else point['index']
            })
        
        return formatted_points
    
    def analyze_with_practical_boxes(self) -> Dict:
        """
        使用实战化箱体策略进行分析
        
        Returns:
            完整的分析结果
        """
        print(f"🔍 开始优化版ZigZag分析 (deviation={self.deviation}%, depth={self.depth})")
        print(f"📊 数据范围: {len(self.data)} 根K线")
        print(f"🎯 识别到 {len(self.swing_points)} 个摆动点")
        
        # 1. 识别关键价格位
        print("\n📍 识别关键价格位...")
        key_levels = self.box_strategy.identify_key_levels(self.swing_points)
        print(f"✅ 识别到 {len(key_levels)} 个关键价格位")
        
        # 2. 识别实战化交易箱体
        print("\n📦 识别实战化交易箱体...")
        trading_boxes = self.box_strategy.identify_trading_boxes(key_levels)
        print(f"✅ 识别到 {len(trading_boxes)} 个交易箱体")
        
        # 3. 检测突破信号
        print("\n🚀 检测突破信号...")
        breakout_signals = self.box_strategy.detect_breakout_signals(trading_boxes)
        print(f"✅ 检测到 {len(breakout_signals)} 个突破信号")
        
        # 4. 模拟交易
        print("\n💰 模拟交易执行...")
        trade_results = self._simulate_trades(breakout_signals)
        
        # 5. 计算统计信息
        stats = self._calculate_statistics(key_levels, trading_boxes, breakout_signals, trade_results)
        
        return {
            'swing_points': self.swing_points,
            'key_levels': key_levels,
            'trading_boxes': trading_boxes,
            'breakout_signals': breakout_signals,
            'trade_results': trade_results,
            'statistics': stats,
            'parameters': {
                'deviation': self.deviation,
                'depth': self.depth,
                'data_length': len(self.data)
            }
        }
    
    def _simulate_trades(self, signals: List[Dict]) -> List[Dict]:
        """
        模拟交易执行
        
        Args:
            signals: 突破信号列表
            
        Returns:
            交易结果列表
        """
        if not signals:
            print("   ⚠️ 无突破信号，跳过交易模拟")
            return []
        
        print(f"   📊 开始模拟 {len(signals)} 个信号的交易...")
        trades = []
        
        for i, signal in enumerate(signals):
            print(f"   处理信号 {i+1}/{len(signals)}...")
            
            entry_index = signal['index']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            signal_type = signal['type']
            
            # 寻找退出点
            exit_info = self._find_exit_point(
                entry_index, entry_price, stop_loss, take_profit, signal_type
            )
            
            if exit_info:
                # 计算交易结果
                if signal_type == 'buy':
                    pnl_pct = (exit_info['exit_price'] - entry_price) / entry_price
                else:  # sell
                    pnl_pct = (entry_price - exit_info['exit_price']) / entry_price
                
                trades.append({
                    'entry_index': entry_index,
                    'entry_price': entry_price,
                    'exit_index': exit_info['exit_index'],
                    'exit_price': exit_info['exit_price'],
                    'exit_reason': exit_info['exit_reason'],
                    'signal_type': signal_type,
                    'pnl_pct': pnl_pct,
                    'pnl_amount': pnl_pct * 10000,  # 假设10000本金
                    'holding_periods': exit_info['exit_index'] - entry_index,
                    'signal_strength': signal['strength'],
                    'box_info': signal.get('box_info', {}),
                    'risk_reward': signal.get('risk_reward', {})
                })
        
        print(f"   ✅ 完成交易模拟，生成 {len(trades)} 笔交易")
        return trades
    
    def _find_exit_point(self, entry_index: int, entry_price: float, 
                        stop_loss: float, take_profit: float, signal_type: str) -> Optional[Dict]:
        """
        寻找交易退出点
        
        Args:
            entry_index: 入场位置
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 止盈价格
            signal_type: 信号类型
            
        Returns:
            退出信息
        """
        max_holding_periods = 50  # 最大持仓周期
        
        for i in range(entry_index + 1, min(len(self.data), entry_index + max_holding_periods + 1)):
            current_high = self.data.loc[i, 'high']
            current_low = self.data.loc[i, 'low']
            current_close = self.data.loc[i, 'close']
            
            if signal_type == 'buy':
                # 多头交易
                if current_high >= take_profit:
                    return {
                        'exit_index': i,
                        'exit_price': take_profit,
                        'exit_reason': 'take_profit'
                    }
                elif current_low <= stop_loss:
                    return {
                        'exit_index': i,
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss'
                    }
            else:
                # 空头交易
                if current_low <= take_profit:
                    return {
                        'exit_index': i,
                        'exit_price': take_profit,
                        'exit_reason': 'take_profit'
                    }
                elif current_high >= stop_loss:
                    return {
                        'exit_index': i,
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss'
                    }
        
        # 超过最大持仓周期，按收盘价退出
        final_index = min(len(self.data) - 1, entry_index + max_holding_periods)
        return {
            'exit_index': final_index,
            'exit_price': self.data.loc[final_index, 'close'],
            'exit_reason': 'max_holding_period'
        }
    
    def _calculate_statistics(self, key_levels: List[Dict], trading_boxes: List[Dict], 
                            signals: List[Dict], trades: List[Dict]) -> Dict:
        """
        计算统计信息
        
        Args:
            key_levels: 关键价格位
            trading_boxes: 交易箱体
            signals: 突破信号
            trades: 交易结果
            
        Returns:
            统计信息
        """
        stats = {
            'swing_points_count': len(self.swing_points),
            'key_levels_count': len(key_levels),
            'trading_boxes_count': len(trading_boxes),
            'signals_count': len(signals),
            'trades_count': len(trades)
        }
        
        if trades:
            pnl_list = [t['pnl_pct'] for t in trades]
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            losing_trades = [t for t in trades if t['pnl_pct'] < 0]
            
            stats.update({
                'total_pnl_pct': sum(pnl_list),
                'avg_pnl_pct': np.mean(pnl_list),
                'win_rate': len(winning_trades) / len(trades),
                'avg_win_pct': np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss_pct': np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0,
                'max_win_pct': max(pnl_list) if pnl_list else 0,
                'max_loss_pct': min(pnl_list) if pnl_list else 0,
                'avg_holding_periods': np.mean([t['holding_periods'] for t in trades]),
                'profit_factor': (sum([t['pnl_pct'] for t in winning_trades]) / 
                                abs(sum([t['pnl_pct'] for t in losing_trades]))) if losing_trades else float('inf')
            })
        else:
            stats.update({
                'total_pnl_pct': 0,
                'avg_pnl_pct': 0,
                'win_rate': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'max_win_pct': 0,
                'max_loss_pct': 0,
                'avg_holding_periods': 0,
                'profit_factor': 0
            })
        
        return stats
    
    def create_detailed_chart(self, analysis_result: Dict, save_path: str = None) -> str:
        """
        创建详细的分析图表
        
        Args:
            analysis_result: 分析结果
            save_path: 保存路径
            
        Returns:
            图表保存路径
        """
        print("📊 开始生成分析图表...")
        
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
        
        print("   绘制价格走势...")
        # 主图：价格和分析结果
        self._plot_price_analysis(ax1, analysis_result)
        
        print("   绘制成交量分析...")
        # 副图：成交量
        self._plot_volume_analysis(ax2, analysis_result)
        
        # 设置标题
        stats = analysis_result['statistics']
        title = (f"优化版ZigZag分析 - 实战化箱体策略\\n"
                f"参数: deviation={self.deviation}%, depth={self.depth} | "
                f"摆动点: {stats['swing_points_count']} | 关键位: {stats['key_levels_count']} | "
                f"交易箱体: {stats['trading_boxes_count']} | 信号: {stats['signals_count']}\\n"
                f"交易统计: 胜率{stats['win_rate']:.1%} | 总收益{stats['total_pnl_pct']:.2%} | "
                f"盈亏比{stats['profit_factor']:.2f}")
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"optimized_zigzag_analysis_{timestamp}.png"
        
        print(f"   保存图表到: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表，不显示
        
        print("✅ 图表生成完成!")
        return save_path
    
    def _plot_price_analysis(self, ax, analysis_result: Dict):
        """绘制价格分析图"""
        # 绘制K线
        for i in range(len(self.data)):
            open_price = self.data.loc[i, 'open']
            high_price = self.data.loc[i, 'high']
            low_price = self.data.loc[i, 'low']
            close_price = self.data.loc[i, 'close']
            
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制影线
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=0.5)
            
            # 绘制实体
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            rect = patches.Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                                   linewidth=0.5, edgecolor='black', 
                                   facecolor=color, alpha=0.7)
            ax.add_patch(rect)
        
        # 绘制ZigZag线
        swing_points = analysis_result['swing_points']
        if len(swing_points) > 1:
            indices = [p['index'] for p in swing_points]
            prices = [p['price'] for p in swing_points]
            ax.plot(indices, prices, 'b-', linewidth=2, alpha=0.8, label='ZigZag线')
        
        # 绘制摆动点
        for point in swing_points:
            color = 'red' if point['type'] == 'high' else 'blue'
            marker = 'v' if point['type'] == 'high' else '^'
            ax.scatter(point['index'], point['price'], color=color, marker=marker, 
                      s=60, zorder=5, alpha=0.8)
        
        # 绘制关键价格位
        key_levels = analysis_result['key_levels']
        for level in key_levels[:5]:  # 只显示前5个最重要的
            price = level['price']
            effectiveness = level['effectiveness']['score']
            
            # 根据有效性设置颜色和透明度
            alpha = 0.3 + effectiveness * 0.4
            color = 'purple'
            
            ax.axhline(y=price, color=color, linestyle='--', alpha=alpha, linewidth=1.5)
            ax.text(len(self.data) * 0.02, price, 
                   f"关键位 {price:.2f} (强度:{level['strength']}, 有效性:{effectiveness:.2f})",
                   fontsize=8, color=color, alpha=0.8)
        
        # 绘制实战化交易箱体
        trading_boxes = analysis_result['trading_boxes']
        colors = ['green', 'orange', 'purple']
        
        for i, box in enumerate(trading_boxes):
            color = colors[i % len(colors)]
            resistance = box['resistance_price']
            support = box['support_price']
            time_range = box['time_range']
            
            # 绘制箱体
            box_width = time_range['extended_end'] - time_range['start_index']
            box_height = resistance - support
            
            rect = patches.Rectangle(
                (time_range['start_index'], support), 
                box_width, box_height,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.15
            )
            ax.add_patch(rect)
            
            # 绘制支撑阻力线
            ax.hlines(resistance, time_range['start_index'], time_range['extended_end'], 
                     colors=color, linestyles='-', linewidth=2, alpha=0.8)
            ax.hlines(support, time_range['start_index'], time_range['extended_end'], 
                     colors=color, linestyles='-', linewidth=2, alpha=0.8)
            
            # 添加箱体信息
            mid_x = (time_range['start_index'] + time_range['extended_end']) / 2
            mid_y = (resistance + support) / 2
            
            box_info = (f"实战箱体{i+1}\\n"
                       f"R: {resistance:.2f}\\n"
                       f"S: {support:.2f}\\n"
                       f"评分: {box['trading_score']:.2f}\\n"
                       f"风险收益: {box['risk_reward_ratio']['ratio']:.1f}")
            
            ax.text(mid_x, mid_y, box_info, fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        # 绘制突破信号
        signals = analysis_result['breakout_signals']
        for signal in signals:
            index = signal['index']
            price = signal['price']
            signal_type = signal['type']
            
            color = 'red' if signal_type == 'buy' else 'blue'
            marker = '^' if signal_type == 'buy' else 'v'
            
            ax.scatter(index, price, color=color, marker=marker, s=100, 
                      zorder=10, edgecolors='white', linewidth=2)
            
            # 添加信号标签
            ax.annotate(f"{signal['signal_type']}\\n强度:{signal['strength']:.2f}", 
                       xy=(index, price), xytext=(10, 20 if signal_type == 'buy' else -20),
                       textcoords='offset points', fontsize=8, color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
        
        ax.set_title("价格走势与实战化箱体分析", fontsize=12, fontweight='bold')
        ax.set_ylabel("价格", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    def _plot_volume_analysis(self, ax, analysis_result: Dict):
        """绘制成交量分析图"""
        # 绘制成交量柱状图
        volumes = self.data['volume'].values
        volume_colors = ['red' if self.data.loc[i, 'close'] >= self.data.loc[i, 'open'] 
                        else 'green' for i in range(len(self.data))]
        
        ax.bar(range(len(volumes)), volumes, color=volume_colors, alpha=0.6, width=0.8)
        
        # 绘制成交量移动平均线
        if hasattr(self.box_strategy.data, 'volume_ma20'):
            ax.plot(range(len(self.data)), self.box_strategy.data['volume_ma20'], 
                   'orange', linewidth=1.5, alpha=0.8, label='成交量MA20')
        
        # 标记突破信号的成交量
        signals = analysis_result['breakout_signals']
        for signal in signals:
            index = signal['index']
            volume = volumes[index]
            
            color = 'red' if signal['type'] == 'buy' else 'blue'
            ax.scatter(index, volume, color=color, s=80, zorder=5, 
                      edgecolors='white', linewidth=1)
        
        ax.set_title("成交量分析", fontsize=12, fontweight='bold')
        ax.set_xlabel("时间", fontsize=10)
        ax.set_ylabel("成交量", fontsize=10)
        ax.grid(True, alpha=0.3)
        if hasattr(self.box_strategy.data, 'volume_ma20'):
            ax.legend()


def main():
    """主函数 - 演示优化版分析器"""
    # 读取测试数据
    try:
        data = pd.read_csv('ETH_USDT_5m.csv')
        print(f"✅ 成功读取数据: {len(data)} 根K线")
        
        # 为了演示效果，只使用最近的5000根K线
        if len(data) > 5000:
            data = data.tail(5000).reset_index(drop=True)
            print(f"📊 为提高处理速度，使用最近 {len(data)} 根K线进行分析")
            
    except FileNotFoundError:
        print("❌ 未找到测试数据文件 ETH_USDT_5m.csv")
        return
    
    # 创建优化版分析器
    analyzer = OptimizedZigZagAnalyzer(data, deviation=1.0, depth=10)
    
    # 执行分析
    result = analyzer.analyze_with_practical_boxes()
    
    # 生成图表
    chart_path = analyzer.create_detailed_chart(result, 
                                               "optimized_zigzag_practical_boxes.png")
    
    print(f"\n📊 分析完成！图表已保存至: {chart_path}")
    
    # 打印详细统计
    stats = result['statistics']
    print("\n" + "="*60)
    print("📈 优化版ZigZag分析统计报告")
    print("="*60)
    print(f"🎯 摆动点数量: {stats['swing_points_count']}")
    print(f"📍 关键价格位: {stats['key_levels_count']}")
    print(f"📦 实战交易箱体: {stats['trading_boxes_count']}")
    print(f"🚀 突破信号: {stats['signals_count']}")
    print(f"💰 模拟交易: {stats['trades_count']}")
    
    if stats['trades_count'] > 0:
        print(f"\n📊 交易表现:")
        print(f"   胜率: {stats['win_rate']:.1%}")
        print(f"   总收益: {stats['total_pnl_pct']:.2%}")
        print(f"   平均收益: {stats['avg_pnl_pct']:.2%}")
        print(f"   最大盈利: {stats['max_win_pct']:.2%}")
        print(f"   最大亏损: {stats['max_loss_pct']:.2%}")
        print(f"   盈亏比: {stats['profit_factor']:.2f}")
        print(f"   平均持仓: {stats['avg_holding_periods']:.1f} 周期")


if __name__ == "__main__":
    main()