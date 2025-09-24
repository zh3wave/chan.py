#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
支撑阻力线分析器 V1.0
专注于识别确定性的支撑阻力线，而非完整箱体
基于ZigZag转折点，识别多次触及且未被有效突破的关键水平线
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import baostock as bs
from scipy.signal import argrelextrema
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SupportResistanceAnalyzer:
    """
    支撑阻力线分析器
    核心思路：基于ZigZag转折点，识别确定性的支撑阻力线
    """
    
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data = None
        self.minute_data = None
        
    def fetch_data(self, start_date: str = "2023-01-01", end_date: str = "2025-01-24"):
        """获取股票数据"""
        print("📊 获取日线数据...")
        
        # 登录系统
        lg = bs.login()
        print("login success!" if lg.error_code == '0' else f"login failed: {lg.error_msg}")
        
        print(f"获取 {self.stock_code} 数据...")
        rs = bs.query_history_k_data_plus(
            self.stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3"
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        # 登出系统
        bs.logout()
        print("logout success!")
        
        if not data_list:
            raise ValueError(f"未获取到 {self.stock_code} 的数据")
        
        # 转换为DataFrame
        self.data = pd.DataFrame(data_list, columns=rs.fields)
        
        # 数据类型转换
        numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.dropna()
        
        print(f"成功获取 {len(self.data)} 条数据")
        return self.data
    
    def zigzag_algorithm(self, highs: np.array, lows: np.array, deviation_pct: float = 3.5):
        """
        ZigZag算法识别价格摆动点
        """
        swing_points = []
        if len(highs) < 3 or len(lows) < 3:
            return swing_points
        
        # 合并高低点数据
        combined_data = []
        for i in range(len(highs)):
            combined_data.append({'index': i, 'high': highs[i], 'low': lows[i]})
        
        # 寻找局部极值
        high_indices = argrelextrema(highs, np.greater, order=2)[0]
        low_indices = argrelextrema(lows, np.less, order=2)[0]
        
        # 合并并排序所有极值点
        all_extrema = []
        for idx in high_indices:
            all_extrema.append({'index': idx, 'price': highs[idx], 'type': 'high'})
        for idx in low_indices:
            all_extrema.append({'index': idx, 'price': lows[idx], 'type': 'low'})
        
        all_extrema.sort(key=lambda x: x['index'])
        
        if len(all_extrema) < 2:
            return swing_points
        
        # 应用ZigZag过滤
        filtered_points = [all_extrema[0]]
        
        for current_point in all_extrema[1:]:
            last_point = filtered_points[-1]
            
            # 计算价格变化百分比
            price_change = abs(current_point['price'] - last_point['price']) / last_point['price']
            
            if price_change >= deviation_pct / 100:
                # 如果类型相同，保留价格更极端的点
                if current_point['type'] == last_point['type']:
                    if ((current_point['type'] == 'high' and current_point['price'] > last_point['price']) or
                        (current_point['type'] == 'low' and current_point['price'] < last_point['price'])):
                        filtered_points[-1] = current_point
                else:
                    filtered_points.append(current_point)
        
        return filtered_points
    
    def identify_key_levels(self, swing_points: List[Dict], price_tolerance: float = 0.015, 
                           min_touches: int = 2) -> List[Dict]:
        """
        识别关键分水线（统一支撑阻力线概念）
        
        Args:
            swing_points: ZigZag转折点
            price_tolerance: 价格容忍度（1.5%）
            min_touches: 最小触及次数（降低到2次）
            
        Returns:
            关键分水线列表
        """
        if len(swing_points) < min_touches:
            return []
        
        # 只考虑最近的转折点（最近50个点或6个月数据）
        recent_points = swing_points[-50:] if len(swing_points) > 50 else swing_points
        
        key_levels = []
        
        # 统一处理所有转折点，不区分高低点
        waterlines = self._find_horizontal_levels(recent_points, price_tolerance, min_touches, 'waterline')
        key_levels.extend(waterlines)
        
        return key_levels
    
    def _find_horizontal_levels(self, points: List[Dict], tolerance: float, 
                               min_touches: int, level_type: str) -> List[Dict]:
        """
        在给定点中寻找分水线 - 基于局部范围（箱体概念）
        """
        if len(points) < min_touches:
            return []
        
        levels = []
        used_points = set()
        
        # 定义局部窗口大小（类似箱体的时间范围控制）
        local_window = min(20, len(points) // 3)  # 最多20个点或总点数的1/3
        
        for i, base_point in enumerate(points):
            if i in used_points:
                continue
            
            # 定义局部搜索范围（箱体范围控制）
            window_start = max(0, i - local_window // 2)
            window_end = min(len(points), i + local_window // 2)
            local_points = points[window_start:window_end]
            
            # 在局部范围内寻找与基准点价格接近的其他点
            similar_points = [base_point]
            similar_indices = [i]
            
            for j in range(window_start, window_end):
                if j == i or j in used_points or j >= len(points):
                    continue
                    
                other_point = points[j]
                price_diff = abs(other_point['price'] - base_point['price']) / base_point['price']
                
                # 在局部范围内，价格接近的点才考虑
                if price_diff <= tolerance:
                    similar_points.append(other_point)
                    similar_indices.append(j)
            
            # 在局部范围内找到足够的触及点即可成立分水线
            if len(similar_points) >= min_touches:
                # 计算平均价格作为分水线价格
                avg_price = np.mean([p['price'] for p in similar_points])
                
                # 分水线的有效范围就是局部范围，不延伸到全局
                start_idx = min([p['index'] for p in similar_points])
                end_idx = max([p['index'] for p in similar_points])
                
                # 限制分水线长度，避免过长的线条
                max_span = 30  # 最大跨度30个数据点
                if end_idx - start_idx > max_span:
                    # 如果跨度太大，缩短到合理范围
                    mid_idx = (start_idx + end_idx) // 2
                    start_idx = max(start_idx, mid_idx - max_span // 2)
                    end_idx = min(end_idx, mid_idx + max_span // 2)
                
                # 计算局部强度（不考虑全局时间衰减）
                local_strength = self._calculate_level_strength(similar_points, tolerance)
                
                level = {
                    'price': avg_price,
                    'type': level_type,
                    'touches': len(similar_points),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'touch_points': similar_points,
                    'strength': local_strength,
                    'local_range': True,  # 标记为局部范围分水线
                    'window_size': window_end - window_start
                }
                
                levels.append(level)
                used_points.update(similar_indices)
        
        # 按强度排序，优先考虑强分水线
        levels.sort(key=lambda x: x['strength'], reverse=True)
        return levels
    
    def _calculate_level_strength(self, touch_points: List[Dict], tolerance: float) -> float:
        """
        计算水平线强度
        考虑因素：触及次数、价格一致性、时间跨度
        """
        touches = len(touch_points)
        
        # 价格一致性（标准差越小越好）
        prices = [p['price'] for p in touch_points]
        price_consistency = 1 / (1 + np.std(prices) / np.mean(prices))
        
        # 时间跨度
        time_span = max([p['index'] for p in touch_points]) - min([p['index'] for p in touch_points])
        time_factor = min(time_span / 100, 2.0)  # 最大2倍加成
        
        strength = touches * price_consistency * (1 + time_factor)
        return strength
    
    def detect_breakout_signals(self, key_levels: List[Dict]) -> List[Dict]:
        """
        检测突破信号 - 先量比放大，再结构突破的逻辑
        """
        if not key_levels or self.data is None:
            return []
        
        signals = []
        prices = self.data['close'].values
        highs = self.data['high'].values
        lows = self.data['low'].values
        volumes = self.data['volume'].values
        dates = self.data['date'].values
        
        # 先识别量比放大的日期（量比 > 1.5）
        volume_surge_dates = []
        for i in range(len(prices)):
            volume_ratio = self._calculate_volume_ratio(volumes, i)
            if volume_ratio > 1.5:  # 量比阈值
                volume_surge_dates.append({
                    'index': i,
                    'date': dates[i],
                    'volume_ratio': volume_ratio,
                    'price': prices[i]
                })
        
        # 对每个量比放大日期，检查是否有结构突破
        for surge_info in volume_surge_dates:
            surge_idx = surge_info['index']
            surge_price = surge_info['price']
            surge_date = surge_info['date']
            
            # 检查前后几天是否有分水线突破
            check_window = 3  # 检查前后3天
            start_idx = max(0, surge_idx - check_window)
            end_idx = min(len(prices) - 1, surge_idx + check_window)
            
            for level in key_levels:
                level_price = level['price']
                level_type = level['type']
                
                # 检查突破条件
                breakout_confirmed = False
                breakout_type = None
                breakout_strength = 0
                
                # 向上突破检查
                if surge_price > level_price * 1.01:  # 突破阈值1%
                    # 确认突破：收盘价站稳在分水线之上
                    days_above = 0
                    for i in range(surge_idx, min(len(prices), surge_idx + 3)):
                        if prices[i] > level_price:
                            days_above += 1
                    
                    if days_above >= 1:  # 至少1天站稳
                        breakout_confirmed = True
                        breakout_type = 'upward'
                        breakout_strength = (surge_price - level_price) / level_price
                
                # 向下突破检查
                elif surge_price < level_price * 0.99:  # 跌破阈值1%
                    # 确认突破：收盘价跌破分水线
                    days_below = 0
                    for i in range(surge_idx, min(len(prices), surge_idx + 3)):
                        if prices[i] < level_price:
                            days_below += 1
                    
                    if days_below >= 1:  # 至少1天跌破
                        breakout_confirmed = True
                        breakout_type = 'downward'
                        breakout_strength = (level_price - surge_price) / level_price
                
                if breakout_confirmed:
                    # 检查是否已经存在相似信号（避免重复）
                    duplicate = False
                    for existing_signal in signals:
                        existing_date = pd.to_datetime(existing_signal['date'])
                        current_date = pd.to_datetime(surge_date)
                        if (abs((existing_date - current_date).days) <= 5 and
                            abs(existing_signal['level_price'] - level_price) < level_price * 0.02):
                            duplicate = True
                            break
                    
                    if not duplicate:
                        signal = {
                            'date': surge_date,
                            'price': surge_price,
                            'level_price': level_price,
                            'level_type': 'waterline',
                            'breakout_type': breakout_type,
                            'strength': breakout_strength,
                            'volume_ratio': surge_info['volume_ratio'],
                            'level_strength': level['strength'],
                            'level_touches': level['touches']
                        }
                        
                        signals.append(signal)
        
        # 按日期排序
        signals.sort(key=lambda x: pd.to_datetime(x['date']))
        
        return signals
    
    def _calculate_volume_ratio(self, volumes: np.array, index: int, period: int = 5) -> float:
        """计算指定位置的量比"""
        if index < period:  # 需要至少5天的历史数据
            return 1.0
            
        # 计算前5日平均成交量
        avg_volume = np.mean(volumes[max(0, index-period):index])
        
        if avg_volume == 0:
            return 1.0
            
        # 当日量比
        current_volume = volumes[index]
        volume_ratio = current_volume / avg_volume
        
        return volume_ratio
    
    def identify_volume_surge_opportunities(self, data):
        """识别量比放大的交易机会 - 量比优先逻辑"""
        opportunities = []
        
        volumes = data['volume'].values
        prices = data['close'].values
        dates = data['date'].values
        
        # 识别所有量比放大的日期
        for i in range(5, len(data)):  # 从第6天开始，确保有足够历史数据
            volume_ratio = self._calculate_volume_ratio(volumes, i)
            
            # 量比阈值分级
            if volume_ratio >= 2.0:  # 强烈放量
                surge_level = 'strong'
            elif volume_ratio >= 1.5:  # 明显放量
                surge_level = 'moderate'
            else:
                continue
                
            opportunities.append({
                'date': dates[i],
                'price': prices[i],
                'volume_ratio': volume_ratio,
                'surge_level': surge_level,
                'index': i
            })
        
        return opportunities
    
    def plot_analysis(self, figsize=(18, 10)):
        """绘制分析图表"""
        if self.data is None:
            print("❌ 请先获取数据")
            return
        
        # 获取数据
        dates = pd.to_datetime(self.data['date'])
        prices = self.data['close'].values
        highs = self.data['high'].values
        lows = self.data['low'].values
        volumes = self.data['volume'].values
        
        # 识别转折点和关键水平线
        swing_points = self.zigzag_algorithm(highs, lows)
        key_levels = self.identify_key_levels(swing_points)
        breakout_signals = self.detect_breakout_signals(key_levels)
        
        print(f"🔍 ZigZag识别到 {len(swing_points)} 个转折点")
        print(f"🎯 识别到 {len(key_levels)} 条关键水平线")
        print(f"📈 检测到 {len(breakout_signals)} 个突破信号")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # 主图：价格和水平线
        ax1.plot(dates, prices, 'b-', linewidth=1, alpha=0.7, label='收盘价')
        ax1.plot(dates, highs, 'g-', linewidth=0.5, alpha=0.5)
        ax1.plot(dates, lows, 'r-', linewidth=0.5, alpha=0.5)
        
        # 绘制ZigZag转折点
        for point in swing_points:
            idx = point['index']
            if idx < len(dates):
                color = 'red' if point['type'] == 'high' else 'green'
                marker = 'v' if point['type'] == 'high' else '^'
                ax1.scatter(dates.iloc[idx], point['price'], 
                           color=color, s=50, marker=marker, zorder=5)
        
        # 绘制关键分水线
        for level in key_levels:
            start_date = dates.iloc[level['start_idx']]
            end_date = dates.iloc[-1]  # 延伸到最后
            
            color = 'purple'  # 统一使用紫色表示分水线
            linestyle = '-' if level['strength'] > 3 else '--'
            alpha = min(0.8, 0.4 + level['strength'] / 8)
            
            ax1.hlines(level['price'], start_date, end_date, 
                      colors=color, linestyles=linestyle, alpha=alpha, linewidth=2)
            
            # 标注分水线信息
            ax1.text(end_date, level['price'], 
                    f"分水线{level['touches']}次", 
                    fontsize=8, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        # 绘制突破信号
        for signal in breakout_signals:
            signal_date = pd.to_datetime(signal['date'])
            if signal_date in dates.values:
                color = 'lime' if signal['breakout_type'] == 'upward' else 'orange'
                marker = '↑' if signal['breakout_type'] == 'upward' else '↓'
                ax1.scatter(signal_date, signal['price'], 
                           color=color, s=100, marker='o', zorder=10)
                ax1.annotate(f"{marker}{signal['strength']:.1%}", 
                           (signal_date, signal['price']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, color=color, weight='bold')
        
        ax1.set_title(f'{self.stock_code} 支撑阻力线分析 (V1.0)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 成交量图
        ax2.bar(dates, volumes, alpha=0.6, color='gray', width=1)
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        
        # 格式化日期轴
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"support_resistance_{self.stock_code}_{timestamp}"
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename}.jpg", dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'swing_points': len(swing_points),
            'key_levels': len(key_levels),
            'breakout_signals': len(breakout_signals),
            'filename': filename
        }
    
    def print_analysis_report(self, key_levels: List[Dict], signals: List[Dict]):
        """打印分析报告"""
        print("\n" + "="*60)
        print(f"📊 {self.stock_code} 支撑阻力线分析报告 V1.0")
        print("="*60)
        
        # 关键分水线统计
        waterlines = [l for l in key_levels if l['type'] == 'waterline']
        
        print(f"\n🎯 关键分水线统计:")
        print(f"   分水线数量: {len(waterlines)}")
        print(f"   总计: {len(key_levels)} 条")
        
        # 突破信号统计
        upward_signals = [s for s in signals if s['breakout_type'] == 'upward']
        downward_signals = [s for s in signals if s['breakout_type'] == 'downward']
        
        print(f"\n📈 突破信号统计:")
        print(f"   向上突破: {len(upward_signals)} ({len(upward_signals)/len(signals)*100:.1f}%)" if signals else "   向上突破: 0")
        print(f"   向下突破: {len(downward_signals)} ({len(downward_signals)/len(signals)*100:.1f}%)" if signals else "   向下突破: 0")
        print(f"   总信号数: {len(signals)}")
        
        # 详细信号列表
        if signals:
            print(f"\n🎯 详细突破信号:")
            for i, signal in enumerate(signals[:10], 1):  # 显示前10个
                date_str = pd.to_datetime(signal['date']).strftime('%Y-%m-%d')
                direction = '📈' if signal['breakout_type'] == 'upward' else '📉'
                level_type = '分水'
                
                print(f"    {i}. {date_str} {direction} "
                      f"突破{level_type}线 价格:{signal['price']:.2f} "
                      f"强度:{signal['strength']:.1%} "
                      f"量比:{signal['volume_ratio']:.1f} "
                      f"线强度:{signal['level_strength']:.1f}")
        
        print("="*60)

def main():
    """主函数"""
    print("🚀 开始支撑阻力线分析...")
    
    # 创建分析器
    analyzer = SupportResistanceAnalyzer("sz.000063")
    
    # 获取数据
    analyzer.fetch_data(start_date="2023-01-01", end_date="2025-01-24")
    
    # 执行分析
    print("\n🔍 执行支撑阻力线分析...")
    
    # 获取基础数据
    highs = analyzer.data['high'].values
    lows = analyzer.data['low'].values
    
    # 识别转折点和关键水平线
    swing_points = analyzer.zigzag_algorithm(highs, lows)
    key_levels = analyzer.identify_key_levels(swing_points)
    breakout_signals = analyzer.detect_breakout_signals(key_levels)
    
    # 打印报告
    analyzer.print_analysis_report(key_levels, breakout_signals)
    
    # 绘制图表
    print("\n📊 生成分析图表...")
    result = analyzer.plot_analysis()
    
    print(f"\n📊 分析结果:")
    print(f"关键转折点数量: {result['swing_points']}")
    print(f"关键分水线数量: {result['key_levels']}")
    print(f"突破信号数量: {result['breakout_signals']}")
    
    print("\n✅ 支撑阻力线分析完成!")
    return analyzer, key_levels, breakout_signals

if __name__ == "__main__":
    analyzer, levels, signals = main()