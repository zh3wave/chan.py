#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZigZag策略可视化分析器 v4.0 - 清晰图表版
生成详细的K线图表，展示：
1. 800根K线的价格走势
2. ZigZag摆动点和箱体
3. 支撑阻力位
4. 买卖信号和交易过程
5. 策略执行结果分析

版本: v4.0 - 清晰图表版
日期: 2025-09-30
备份: versions/zigzag_visual_analyzer_v4.0_清晰图表版_20250930_110605.py

主要特性:
✅ 真正的时效性权重：完全放弃遥远价位（前40%历史数据）
✅ 清晰简洁的图表设计：按用户示例图样式绘制
✅ 优化箱体显示：最多5个箱体，按时间顺序，最新箱体绿色标识
✅ 简化颜色方案：淡色背景，优化ZigZag摆动点颜色
✅ 移除冗余信息：简化标注，突出关键价格区间

改进要点:
1. 支撑阻力位识别限制在最近60%数据点
2. 箱体识别按时效性评分，越新越重要
3. 图表信息密度大幅降低，视觉清晰度显著提升
4. 颜色方案优化，符合短线交易需求
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_zigzag_standard import OKXZigZag

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class ZigZagVisualAnalyzer:
    """ZigZag策略可视化分析器"""
    
    def __init__(self):
        self.data = None
        self.swing_points = []
        self.levels = []
        self.signals = []
        self.trades = []
        
    def load_data_segment(self, file_path: str, start_index: int = 200000, length: int = 1600) -> bool:
        """加载指定段的数据"""
        try:
            # 读取完整数据
            full_data = pd.read_csv(file_path)
            print(f"✅ 成功加载数据，总共 {len(full_data)} 条记录")
            
            # 选择指定段的数据
            end_index = min(start_index + length, len(full_data))
            self.data = full_data.iloc[start_index:end_index].copy()
            
            # 转换日期格式
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.reset_index(drop=True)
            
            print(f"📊 分析数据段: 第 {start_index} 到 {end_index-1} 条记录 ({len(self.data)} 根K线)")
            print(f"📅 时间范围: {self.data['date'].iloc[0]} 到 {self.data['date'].iloc[-1]}")
            print(f"💰 价格范围: ${self.data['low'].min():.2f} - ${self.data['high'].max():.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
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
    
    def calculate_volume_ratio(self, volumes: np.array, period: int = 5) -> np.array:
        """计算量比指标"""
        volume_ratio = np.ones_like(volumes, dtype=float)
        
        for i in range(period, len(volumes)):
            avg_volume = np.mean(volumes[i-period:i])
            if avg_volume > 0:
                volume_ratio[i] = volumes[i] / avg_volume
        
        return volume_ratio
    
    def analyze_with_zigzag(self, deviation: float = 1.0, depth: int = 10):
        """使用ZigZag参数进行分析"""
        prices = self.data['close'].values
        highs = self.data['high'].values
        lows = self.data['low'].values
        volumes = self.data['volume'].values
        
        # 计算技术指标（仅保留成交量）
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        # 使用OKX ZigZag算法
        zigzag = OKXZigZag(deviation=deviation, depth=depth)
        self.swing_points, zigzag_line = zigzag.calculate(highs, lows)
        
        print(f"🔍 ZigZag分析 (deviation={deviation}%, depth={depth}):")
        print(f"   识别摆动点: {len(self.swing_points)} 个")
        
        # 识别支撑阻力位
        self.levels = self.identify_support_resistance(self.swing_points)
        print(f"   支撑阻力位: {len(self.levels)} 个")
        
        # 检测交易信号（移除MACD参数）
        self.signals = self.detect_breakout_signals(self.levels, volume_ratio)
        print(f"   交易信号: {len(self.signals)} 个")
        
        # 执行模拟交易
        self.trades = self.simulate_trades(self.signals)
        print(f"   完成交易: {len(self.trades)} 笔")
        
        return {
            'swing_points': len(self.swing_points),
            'levels': len(self.levels),
            'signals': len(self.signals),
            'trades': len(self.trades)
        }
    
    def identify_support_resistance(self, swing_points: List[Dict], price_tolerance: float = 0.008) -> List[Dict]:
        """识别支撑阻力位和箱体 - 时效性优先，放弃遥远价位的影响"""
        if len(swing_points) < 2:
            return []
        
        levels = []
        processed_points = set()
        total_points = len(swing_points)
        
        # 定义时效性范围：只考虑最近的60%数据点
        effective_start = max(0, int(total_points * 0.4))  # 放弃前40%的历史数据
        effective_points = swing_points[effective_start:]
        
        # 第一步：识别精确的水平线（仅在有效时间范围内）
        for i, point1 in enumerate(effective_points):
            if i in processed_points:
                continue
                
            cluster = [point1]
            cluster_indices = {i}
            
            # 在有效范围内寻找精确匹配的点
            for j, point2 in enumerate(effective_points):
                if i != j and j not in processed_points:
                    price_diff = abs(point1['price'] - point2['price']) / point1['price']
                    if price_diff <= price_tolerance:
                        cluster.append(point2)
                        cluster_indices.add(j)
            
            # 如果找到至少2个相近的点，形成水平线
            if len(cluster) >= 2:
                # 计算更精确的价格（使用中位数）
                prices = [p['price'] for p in cluster]
                precise_price = np.median(prices)
                
                min_index = min([p['index'] for p in cluster])
                max_index = max([p['index'] for p in cluster])
                
                # 分析水平线的性质
                high_count = sum(1 for p in cluster if p['type'] == 'high')
                low_count = sum(1 for p in cluster if p['type'] == 'low')
                
                # 计算时间跨度
                time_span = max_index - min_index
                
                # 确定水平线类型
                if high_count > low_count:
                    line_type = 'resistance_dominant'
                elif low_count > high_count:
                    line_type = 'support_dominant'
                else:
                    line_type = 'neutral'
                
                # 重要性评分：强调最近性，不考虑历史权重
                position_in_effective = (max_index - effective_start) / len(effective_points)
                recency_bonus = 1.0 + position_in_effective * 0.5  # 最新的点获得更高权重
                importance_score = len(cluster) * recency_bonus
                
                levels.append({
                    'type': 'horizontal_line',
                    'dominant_role': line_type,
                    'price': precise_price,
                    'strength': len(cluster),
                    'importance': importance_score,
                    'start_index': min_index,
                    'end_index': max_index,
                    'time_span': time_span,
                    'points': cluster,
                    'high_points': high_count,
                    'low_points': low_count,
                    'effective_range': True,  # 标记为有效范围内的水平线
                    'point_details': [f"Point {p['index']+1}({p['type']})" for p in sorted(cluster, key=lambda x: x['index'])],
                    'recency': position_in_effective
                })
                
                processed_points.update(cluster_indices)
        
        # 第二步：识别箱体区间（仅在有效时间范围内）
        boxes = self.identify_price_boxes(effective_points, levels)
        
        # 合并水平线和箱体，按重要性排序
        all_levels = levels + boxes
        
        return sorted(all_levels, key=lambda x: x.get('importance', x['strength']), reverse=True)
    
    def identify_price_boxes(self, swing_points: List[Dict], levels: List[Dict]) -> List[Dict]:
        """识别有生命周期的价格箱体 - 按时效性权重，最多识别5个箱体"""
        boxes = []
        total_points = len(swing_points)
        
        # 定义时间窗口大小（根据数据量动态调整）
        window_size = max(8, min(15, total_points // 8))  # 8-15个点为一个窗口
        
        # 从最新的数据开始向前分析（时效性优先）
        analyzed_indices = set()
        
        # 最多识别5个箱体，从最新开始
        for box_count in range(5):
            best_box = None
            best_score = 0
            
            # 从最新数据开始扫描
            for i in range(max(0, total_points - window_size), -1, -1):
                if i + window_size > total_points:
                    continue
                    
                window_points = swing_points[i:i + window_size]
                
                # 跳过已经被分析过的区间
                if any(p['index'] in analyzed_indices for p in window_points):
                    continue
                
                # 计算价格范围
                prices = [p['price'] for p in window_points]
                price_min = min(prices)
                price_max = max(prices)
                price_range = price_max - price_min
                
                # 检查是否形成有效箱体
                if price_range / price_min <= 0.06:  # 6%的价格震荡范围
                    # 计算箱体的上下边界
                    resistance_level = price_max
                    support_level = price_min
                    
                    # 检查箱体内的价格行为
                    highs_near_top = sum(1 for p in window_points 
                                       if p['type'] == 'high' and 
                                       abs(p['price'] - resistance_level) / resistance_level <= 0.015)
                    lows_near_bottom = sum(1 for p in window_points 
                                         if p['type'] == 'low' and 
                                         abs(p['price'] - support_level) / support_level <= 0.015)
                    
                    # 如果有足够的测试点，计算箱体评分
                    if highs_near_top >= 2 and lows_near_bottom >= 2:
                        # 时效性评分：越新的箱体评分越高
                        recency_score = (i + window_size) / total_points
                        
                        # 强度评分：测试次数越多评分越高
                        strength_score = (highs_near_top + lows_near_bottom) / 6
                        
                        # 综合评分
                        total_score = recency_score * 0.7 + strength_score * 0.3
                        
                        if total_score > best_score:
                            best_score = total_score
                            box_start = window_points[0]['index']
                            box_end = window_points[-1]['index']
                            
                            best_box = {
                                'type': 'price_box',
                                'dominant_role': 'box_range',
                                'resistance_price': resistance_level,
                                'support_price': support_level,
                                'center_price': (resistance_level + support_level) / 2,
                                'price': (resistance_level + support_level) / 2,
                                'strength': highs_near_top + lows_near_bottom,
                                'importance': total_score * 10,  # 标准化重要性
                                'start_index': box_start,
                                'end_index': box_end,
                                'time_span': box_end - box_start,
                                'box_height': price_range,
                                'box_height_pct': price_range / support_level,
                                'resistance_tests': highs_near_top,
                                'support_tests': lows_near_bottom,
                                'points': window_points,
                                'recency_rank': box_count + 1,  # 时间排序
                                'is_latest': box_count == 0,  # 是否为最新箱体
                                'point_details': [f"Box{box_count+1}({box_start+1}-{box_end+1}): R{resistance_level:.2f} S{support_level:.2f}"]
                            }
            
            # 如果找到了有效箱体，添加到结果中
            if best_box:
                boxes.append(best_box)
                # 标记已分析的点
                for p in best_box['points']:
                    analyzed_indices.add(p['index'])
            else:
                break  # 没有找到更多有效箱体
        
        return boxes
    
    def detect_breakout_signals(self, levels: List[Dict], volume_ratio: np.array) -> List[Dict]:
        """检测突破信号 - 支持水平线和箱体突破"""
        signals = []
        
        if not levels:
            return signals
        
        prices = self.data['close'].values
        volumes = self.data['volume'].values
        
        for i in range(50, len(prices)):  # 从第50根K线开始检测
            current_price = prices[i]
            current_volume = volumes[i]
            
            # 检查每个水平线和箱体
            for level in levels:
                if level['type'] == 'horizontal_line':
                    # 水平线突破逻辑
                    level_price = level['price']
                    dominant_role = level['dominant_role']
                    importance = level.get('importance', level['strength'])
                    
                    # 向上突破水平线
                    if (current_price > level_price * 1.005 and  # 突破0.5%
                        prices[i-1] <= level_price):
                        
                        # 成交量确认
                        volume_confirm = volume_ratio[i] > 1.2
                        
                        # 根据水平线性质和重要性调整信号强度
                        base_strength = 0.7
                        if dominant_role == 'resistance_dominant':
                            base_strength = 0.8  # 突破阻力位更强
                        elif dominant_role == 'support_dominant':
                            base_strength = 0.6  # 突破支撑位转阻力
                        
                        signal_strength = base_strength
                        if volume_confirm:
                            signal_strength += 0.3
                        
                        # 根据重要性调整
                        signal_strength += min(0.3, importance * 0.05)
                        
                        signals.append({
                            'index': i,
                            'type': 'buy',
                            'price': current_price,
                            'level_price': level_price,
                            'level_role': dominant_role,
                            'strength': signal_strength,
                            'volume_confirm': volume_confirm,
                            'stop_loss': level_price * 0.98,  # 2%止损
                            'take_profit': current_price * 1.06,  # 6%止盈
                            'signal_source': 'horizontal_line'
                        })
                    
                    # 向下突破水平线
                    elif (current_price < level_price * 0.995 and  # 突破0.5%
                          prices[i-1] >= level_price):
                        
                        # 成交量确认
                        volume_confirm = volume_ratio[i] > 1.2
                        
                        # 根据水平线性质调整信号强度
                        base_strength = 0.7
                        if dominant_role == 'support_dominant':
                            base_strength = 0.8  # 突破支撑位更强
                        elif dominant_role == 'resistance_dominant':
                            base_strength = 0.6  # 突破阻力位转支撑
                        
                        signal_strength = base_strength
                        if volume_confirm:
                            signal_strength += 0.3
                        
                        # 根据重要性调整
                        signal_strength += min(0.3, importance * 0.05)
                        
                        signals.append({
                            'index': i,
                            'type': 'sell',
                            'price': current_price,
                            'level_price': level_price,
                            'level_role': dominant_role,
                            'strength': signal_strength,
                            'volume_confirm': volume_confirm,
                            'stop_loss': level_price * 1.02,  # 2%止损
                            'take_profit': current_price * 0.94,  # 6%止盈
                            'signal_source': 'horizontal_line'
                        })
                
                elif level['type'] == 'price_box':
                    # 箱体突破逻辑
                    resistance = level['resistance_price']
                    support = level['support_price']
                    box_start = level['start_index']
                    box_end = level['end_index']
                    
                    # 只在箱体生命周期内或刚结束后检测突破
                    if i >= box_start and i <= box_end + 20:
                        
                        # 向上突破箱体阻力位
                        if (current_price > resistance * 1.008 and  # 突破0.8%
                            prices[i-1] <= resistance):
                            
                            volume_confirm = volume_ratio[i] > 1.5  # 箱体突破需要更强成交量
                            
                            # 箱体突破信号强度
                            signal_strength = 0.9  # 箱体突破通常更可靠
                            if volume_confirm:
                                signal_strength += 0.4
                            
                            # 根据箱体测试次数调整
                            signal_strength += min(0.2, level['resistance_tests'] * 0.05)
                            
                            signals.append({
                                'index': i,
                                'type': 'buy',
                                'price': current_price,
                                'level_price': resistance,
                                'level_role': 'box_resistance_breakout',
                                'strength': signal_strength,
                                'volume_confirm': volume_confirm,
                                'stop_loss': support,  # 止损设在箱体支撑位
                                'take_profit': current_price + (resistance - support) * 1.5,  # 目标为箱体高度的1.5倍
                                'signal_source': 'price_box',
                                'box_info': f"Box({box_start+1}-{box_end+1})"
                            })
                        
                        # 向下突破箱体支撑位
                        elif (current_price < support * 0.992 and  # 突破0.8%
                              prices[i-1] >= support):
                            
                            volume_confirm = volume_ratio[i] > 1.5
                            
                            # 箱体突破信号强度
                            signal_strength = 0.9
                            if volume_confirm:
                                signal_strength += 0.4
                            
                            # 根据箱体测试次数调整
                            signal_strength += min(0.2, level['support_tests'] * 0.05)
                            
                            signals.append({
                                'index': i,
                                'type': 'sell',
                                'price': current_price,
                                'level_price': support,
                                'level_role': 'box_support_breakout',
                                'strength': signal_strength,
                                'volume_confirm': volume_confirm,
                                'stop_loss': resistance,  # 止损设在箱体阻力位
                                'take_profit': current_price - (resistance - support) * 1.5,  # 目标为箱体高度的1.5倍
                                'signal_source': 'price_box',
                                'box_info': f"Box({box_start+1}-{box_end+1})"
                            })
        
        return signals
    
    def simulate_trades(self, signals: List[Dict]) -> List[Dict]:
        """模拟交易执行"""
        if not signals:
            return []
        
        prices = self.data['close'].values
        trades = []
        
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
            
            trades.append({
                'entry_index': entry_index,
                'exit_index': exit_index,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'signal_type': signal_type,
                'return_pct': return_pct,
                'exit_reason': exit_reason,
                'strength': signal['strength'],
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
        
        return trades
    
    def create_comprehensive_chart(self, config_name: str):
        """创建综合分析图表"""
        fig = plt.figure(figsize=(20, 16))
        
        # 创建子图布局 (3行1列，比例为3:1:1)
        gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 0, 0], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])  # 主图：K线图 (占3个单位高度)
        ax2 = fig.add_subplot(gs[1])  # 副图1：成交量 (占1个单位高度)
        ax3 = fig.add_subplot(gs[2])  # 副图2：量比 (占1个单位高度)
        
        # 主图：K线图
        self.plot_candlestick_chart(ax1, f" - {config_name}")
        
        # 副图1：成交量
        self.plot_volume_chart(ax2)
        
        # 副图2：量比
        self.plot_volume_ratio_chart(ax3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = f'zigzag_detailed_analysis_{config_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_detailed_chart(self, title_suffix: str = "", save_path: str = None):
        """创建详细的K线图表"""
        fig = plt.figure(figsize=(20, 16))
        
        # 创建子图布局 (3行1列，比例为3:1:1)
        gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 0, 0], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])  # 主图：K线图 (占3个单位高度)
        ax2 = fig.add_subplot(gs[1])  # 副图1：成交量 (占1个单位高度)
        ax3 = fig.add_subplot(gs[2])  # 副图2：量比 (占1个单位高度)
        
        # 主图：K线图
        self.plot_candlestick_chart(ax1, title_suffix)
        
        # 副图1：成交量
        self.plot_volume_chart(ax2)
        
        # 副图2：量比
        self.plot_volume_ratio_chart(ax3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 详细图表已保存: {save_path}")
        
        plt.show()
    
    def plot_candlestick_chart(self, ax, title_suffix: str = ""):
        """绘制K线图"""
        # 准备数据
        dates = self.data['date']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        closes = self.data['close']
        
        # 绘制K线
        for i in range(len(self.data)):
            color = 'red' if closes.iloc[i] >= opens.iloc[i] else 'green'
            
            # 绘制影线
            ax.plot([i, i], [lows.iloc[i], highs.iloc[i]], color='black', linewidth=0.5)
            
            # 绘制实体
            body_height = abs(closes.iloc[i] - opens.iloc[i])
            body_bottom = min(opens.iloc[i], closes.iloc[i])
            
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # 绘制ZigZag摆动点（使用淡色方案）
        if self.swing_points:
            for point in self.swing_points:
                # 使用更淡的颜色
                color = 'lightcoral' if point['type'] == 'high' else 'lightblue'
                marker = '^' if point['type'] == 'high' else 'v'
                ax.scatter(point['index'], point['price'], color=color, s=80, 
                          marker=marker, zorder=5, edgecolors='gray', linewidth=0.8, alpha=0.8)
                
                # 简化标注，只显示价格
                ax.annotate(f"{point['price']:.1f}", 
                           (point['index'], point['price']), 
                           xytext=(3, 8 if point['type'] == 'high' else -12),
                           textcoords='offset points', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.6, edgecolor='none'))
        
        # 绘制ZigZag连线（使用更淡的颜色）
        if len(self.swing_points) > 1:
            swing_x = [p['index'] for p in self.swing_points]
            swing_y = [p['price'] for p in self.swing_points]
            ax.plot(swing_x, swing_y, 'mediumpurple', linewidth=1.5, alpha=0.6, label='ZigZag')
        
        # 绘制支撑阻力位和箱体（清晰简洁的显示）
        box_count = 0
        for level in self.levels:
            if level['type'] == 'horizontal_line':
                # 只显示最重要的水平线（前3个）
                if level.get('importance', 0) < sorted([l.get('importance', 0) for l in self.levels if l['type'] == 'horizontal_line'], reverse=True)[min(2, len([l for l in self.levels if l['type'] == 'horizontal_line'])-1)]:
                    continue
                    
                # 绘制水平线（简化样式）
                line_width = 2 if level.get('importance', 1) > 5 else 1.5
                
                # 简化颜色方案
                if level['dominant_role'] == 'resistance_dominant':
                    color = 'red'
                    role_text = 'R'
                elif level['dominant_role'] == 'support_dominant':
                    color = 'green'
                    role_text = 'S'
                else:
                    color = 'gray'
                    role_text = 'N'
                
                # 绘制水平线
                ax.axhline(y=level['price'], color=color, linestyle='-', 
                          linewidth=line_width, alpha=0.8, zorder=3)
                
                # 简化标注
                ax.text(len(self.data) * 0.02, level['price'], 
                       f"{role_text} ${level['price']:.1f}", 
                       fontsize=9, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
            
            elif level['type'] == 'price_box' and box_count < 5:
                # 按时间顺序显示最多5个箱体
                box_count += 1
                
                # 最新箱体用绿色，其他用蓝色
                if level.get('is_latest', False):
                    box_color = 'limegreen'
                    box_alpha = 0.15
                    edge_color = 'green'
                    edge_width = 2
                else:
                    box_color = 'lightblue'
                    box_alpha = 0.1
                    edge_color = 'blue'
                    edge_width = 1.5
                
                # 绘制箱体矩形
                start_x = level['start_index']
                end_x = level['end_index']
                box_width = end_x - start_x
                box_height = level['resistance_price'] - level['support_price']
                
                rect = Rectangle((start_x, level['support_price']), box_width, box_height,
                               facecolor=box_color, alpha=box_alpha, 
                               edgecolor=edge_color, linewidth=edge_width, zorder=2)
                ax.add_patch(rect)
                
                # 绘制箱体边界线
                ax.axhline(y=level['resistance_price'], xmin=start_x/len(self.data), xmax=end_x/len(self.data),
                          color=edge_color, linestyle='-', linewidth=edge_width, alpha=0.8)
                ax.axhline(y=level['support_price'], xmin=start_x/len(self.data), xmax=end_x/len(self.data),
                          color=edge_color, linestyle='-', linewidth=edge_width, alpha=0.8)
                
                # 简化箱体标注
                box_label = f"Box{box_count}"
                if level.get('is_latest', False):
                    box_label += " (Latest)"
                
                ax.text(start_x + box_width * 0.1, level['support_price'] + box_height * 0.1,
                       box_label, fontsize=8, color=edge_color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=edge_color))
        
        # 设置图表标题和标签
        # 设置图表标题和标签
        ax.set_title(f'ZigZag Strategy Analysis - 600 Candlesticks{title_suffix}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 绘制交易信号（简化显示）
        for signal in self.signals:
            color = 'lime' if signal['type'] == 'buy' else 'red'
            marker = '^' if signal['type'] == 'buy' else 'v'
            
            ax.scatter(signal['index'], signal['price'], color=color, s=150, 
                      marker=marker, zorder=10, edgecolors='black', linewidth=1.5, alpha=0.9)
            
            # 简化信号标注
            signal_text = f"{'Buy' if signal['type'] == 'buy' else 'Sell'} ${signal['price']:.1f}"
            ax.annotate(signal_text, 
                       (signal['index'], signal['price']), 
                       xytext=(8, 15 if signal['type'] == 'buy' else -20),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='none'),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
        
        # 绘制交易过程（简化显示）
        for i, trade in enumerate(self.trades):
            entry_idx = trade['entry_index']
            exit_idx = trade['exit_index']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            
            # 简化交易区间背景
            profit_color = 'lightgreen' if trade['return_pct'] > 0 else 'lightcoral'
            ax.axvspan(entry_idx, exit_idx, alpha=0.15, color=profit_color)
            
            # 出场点标记
            exit_color = 'green' if trade['return_pct'] > 0 else 'red'
            ax.scatter(exit_idx, exit_price, color=exit_color, s=120, 
                      marker='x', zorder=10, linewidth=2, alpha=0.9)
        
        # 设置x轴标签（简化）
        dates = pd.to_datetime(self.data.index)
        step = max(1, len(self.data) // 8)
        ax.set_xticks(range(0, len(self.data), step))
        ax.set_xticklabels([dates[i].strftime('%m-%d') for i in range(0, len(self.data), step)], 
                          rotation=45, fontsize=10)
    
    def plot_volume_chart(self, ax):
        """绘制成交量图"""
        volumes = self.data['volume']
        colors = ['red' if self.data['close'].iloc[i] >= self.data['open'].iloc[i] else 'green' 
                 for i in range(len(self.data))]
        
        ax.bar(range(len(volumes)), volumes, color=colors, alpha=0.7)
        ax.set_ylabel('Volume', fontsize=12)
        ax.set_title('Volume Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    def plot_volume_ratio_chart(self, ax):
        """绘制量比图"""
        volumes = self.data['volume'].values
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        # 绘制量比线
        ax.plot(range(len(volume_ratio)), volume_ratio, color='purple', linewidth=1.5, label='Volume Ratio')
        
        # 添加基准线
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Baseline')
        ax.axhline(y=1.2, color='red', linestyle='--', alpha=0.7, label='High Volume')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Low Volume')
        
        # 填充区域
        ax.fill_between(range(len(volume_ratio)), volume_ratio, 1.0, 
                       where=(volume_ratio > 1.0), color='red', alpha=0.2)
        ax.fill_between(range(len(volume_ratio)), volume_ratio, 1.0, 
                       where=(volume_ratio < 1.0), color='green', alpha=0.2)
        
        ax.set_ylabel('Volume Ratio', fontsize=12)
        ax.set_title('Volume Ratio Indicator', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

def main():
    """主函数"""
    print("🚀 ZigZag策略可视化分析器启动")
    
    # 初始化分析器
    analyzer = ZigZagVisualAnalyzer()
    
    # 加载ETH数据
    eth_file = "../ETH_USDT_5m.csv"
    if not analyzer.load_data_segment(eth_file, start_index=200000, length=1600):
        return
    
    # 测试不同参数配置
    test_configs = [
        {'name': 'customer_recommended', 'deviation': 1.0, 'depth': 10}
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"📊 分析配置: {config['name']}")
        print(f"   参数: deviation={config['deviation']}%, depth={config['depth']}")
        print('='*60)
        
        # 执行分析
        result = analyzer.analyze_with_zigzag(config['deviation'], config['depth'])
        
        # 生成详细图表
        title_suffix = f" ({config['name']})"
        save_path = f"zigzag_detailed_analysis_{config['name'].replace(' ', '_')}.png"
        analyzer.create_detailed_chart(title_suffix, save_path)
        
        # 打印分析结果
        print(f"\n📈 分析结果:")
        print(f"   摆动点: {result['swing_points']} 个")
        print(f"   支撑阻力位: {result['levels']} 个")
        print(f"   交易信号: {result['signals']} 个")
        print(f"   完成交易: {result['trades']} 笔")
        
        if analyzer.trades:
            total_return = sum([t['return_pct'] for t in analyzer.trades])
            win_trades = [t for t in analyzer.trades if t['return_pct'] > 0]
            win_rate = len(win_trades) / len(analyzer.trades) if analyzer.trades else 0
            
            print(f"   总收益率: {total_return:.2%}")
            print(f"   胜率: {win_rate:.1%}")
            print(f"   平均单笔收益: {total_return/len(analyzer.trades):.2%}")
        
        print(f"   图表已保存: {save_path}")
    
    print("\n✅ 可视化分析完成！")
    print("💡 通过图表可以清楚看到:")
    print("   - ZigZag摆动点识别是否准确")
    print("   - 支撑阻力位是否有效")
    print("   - 交易信号的时机是否合适")
    print("   - 止损止盈设置是否合理")
    print("   - 策略的优化空间在哪里")

if __name__ == "__main__":
    main()