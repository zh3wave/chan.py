#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
进阶版箱体突破分析器 (Enhanced Analyzer Advanced)
融合Enhanced和Simple逻辑的混合箱体识别系统

核心特性：
1. 优先使用Enhanced逻辑识别箱体
2. 箱体边界基于ZigZag点确定，确保清晰边界
3. Enhanced未检测到时，补充Simple逻辑识别的箱体
4. 冲突处理：Enhanced优先，但边界按ZigZag确定
5. 保持max_volatility=0.12的约束条件
6. 重叠箱体聚合功能，使用Simple边界

作者: AI Assistant
版本: V3.1
日期: 2025-09-25
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedBoxBreakoutAnalyzer:
    """进阶版箱体突破分析器"""
    
    def __init__(self, stock_code: str = "sz.000063"):
        self.stock_code = stock_code
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.array, np.array, np.array, np.array]:
        """数据预处理"""
        # 转换为数值类型并处理缺失值
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna(subset=numeric_columns)
        
        prices = data['close'].values
        volumes = data['volume'].values
        highs = data['high'].values
        lows = data['low'].values
        
        return prices, volumes, highs, lows
    
    def calculate_macd(self, prices: np.array, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> dict:
        """计算MACD指标"""
        # 计算EMA
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_values = np.zeros_like(data)
            ema_values[0] = data[0]
            for i in range(1, len(data)):
                ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
            return ema_values
        
        fast_ema = ema(prices, fast_period)
        slow_ema = ema(prices, slow_period)
        
        # DIF线
        dif = fast_ema - slow_ema
        
        # DEA线（信号线）
        dea = ema(dif, signal_period)
        
        # MACD柱状图
        macd = (dif - dea) * 2
        
        return {
            'dif': dif,
            'dea': dea,
            'macd': macd
        }
    
    def calculate_volume_ratio(self, volumes: np.array, period: int = 5) -> np.array:
        """计算量比指标"""
        volume_ratio = np.ones_like(volumes, dtype=float)
        
        for i in range(period, len(volumes)):
            avg_volume = np.mean(volumes[i-period:i])
            if avg_volume > 0:
                volume_ratio[i] = volumes[i] / avg_volume
        
        return volume_ratio
    
    def calculate_fibonacci_levels(self, high: float, low: float) -> dict:
        """计算斐波那契回调位和扩展位"""
        diff = high - low
        
        # 回调位（从高点向下）
        retracement_levels = {
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '61.8%': high - diff * 0.618
        }
        
        # 扩展位（突破后的目标位）
        extension_levels = {
            '127.2%': low + diff * 1.272,
            '161.8%': low + diff * 1.618,
            '261.8%': low + diff * 2.618
        }
        
        return {
            'retracement': retracement_levels,
            'extension': extension_levels
        }
    
    def zigzag_algorithm(self, highs: np.array, lows: np.array, deviation_pct: float = 3.5):
        """
        ZigZag算法识别价格摆动点，使用最高价和最低价
        优化版本：更精确地识别真实的高低点
        """
        swing_points = []
        zigzag_line = []
        
        if len(highs) < 3 or len(lows) < 3:
            return swing_points, zigzag_line
        
        # 寻找局部极值点
        high_indices = argrelextrema(highs, np.greater, order=3)[0]
        low_indices = argrelextrema(lows, np.less, order=3)[0]
        
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
            
            if price_change_pct >= deviation_pct:
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
    
    def identify_boxes_enhanced(self, prices: np.array, volumes: np.array, 
                               macd_data: dict, min_duration: int = 15, 
                               max_volatility: float = 0.12):
        """
        Enhanced版箱体识别算法
        结合MACD和成交量特征进行验证
        """
        boxes = []
        window_size = min_duration
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        for i in range(window_size, len(prices), 5):
            window_prices = prices[i-window_size:i]
            window_volumes = volumes[i-window_size:i]
            window_macd = macd_data['macd'][i-window_size:i]
            window_vol_ratio = volume_ratio[i-window_size:i]
            
            window_high = np.max(window_prices)
            window_low = np.min(window_prices)
            window_mean = np.mean(window_prices)
            
            # 基础箱体条件
            if window_mean > 0:
                volatility = (window_high - window_low) / window_mean
            else:
                continue
            
            if volatility <= max_volatility:
                # 计算触及次数
                tolerance = (window_high - window_low) * 0.08
                
                upper_touches = sum(1 for p in window_prices if abs(p - window_high) <= tolerance)
                lower_touches = sum(1 for p in window_prices if abs(p - window_low) <= tolerance)
                
                if upper_touches >= 2 and lower_touches >= 2:
                    # MACD验证：箱体内MACD应该相对平稳
                    macd_volatility = np.std(window_macd) / (np.mean(np.abs(window_macd)) + 1e-6)
                    
                    # 成交量验证：箱体内成交量相对稳定
                    volume_stability = np.std(window_vol_ratio) / (np.mean(window_vol_ratio) + 1e-6)
                    
                    # 计算箱体强度（综合评分）
                    base_strength = upper_touches + lower_touches
                    macd_score = max(0, 5 - macd_volatility * 10)  # MACD越稳定分数越高
                    volume_score = max(0, 3 - volume_stability * 5)  # 成交量越稳定分数越高
                    
                    total_strength = base_strength + macd_score + volume_score
                    
                    box = {
                        'start_idx': i - window_size,
                        'end_idx': i - 1,
                        'resistance': window_high,
                        'support': window_low,
                        'duration': window_size,
                        'volatility': volatility,
                        'strength': total_strength,
                        'macd_score': macd_score,
                        'volume_score': volume_score,
                        'source': 'enhanced'
                    }
                    
                    boxes.append(box)
        
        return self._filter_overlapping_boxes(boxes)
    
    def identify_boxes_simple(self, prices: np.array, volumes: np.array, 
                             highs: np.array, lows: np.array,
                             min_duration: int = 15, max_volatility: float = 0.12):
        """
        Simple版箱体识别算法（基于ZigZag）
        调整参数以符合进阶版要求
        """
        swing_points, _ = self.zigzag_algorithm(highs, lows, deviation_pct=3.5)
        boxes = []
        
        if len(swing_points) < 4:
            return boxes
        
        # 寻找箱体模式
        for i in range(len(swing_points) - 3):
            # 获取连续的4个摆动点
            points = swing_points[i:i+4]
            
            # 检查是否形成箱体模式（高-低-高-低 或 低-高-低-高）
            if (points[0]['type'] == 'high' and points[1]['type'] == 'low' and 
                points[2]['type'] == 'high' and points[3]['type'] == 'low'):
                
                # 计算箱体参数
                resistance = max(points[0]['price'], points[2]['price'])
                support = min(points[1]['price'], points[3]['price'])
                
            elif (points[0]['type'] == 'low' and points[1]['type'] == 'high' and 
                  points[2]['type'] == 'low' and points[3]['type'] == 'high'):
                
                resistance = max(points[1]['price'], points[3]['price'])
                support = min(points[0]['price'], points[2]['price'])
                
            else:
                continue
            
            # 验证箱体有效性
            if resistance <= support:
                continue
            
            start_idx = points[0]['index']
            end_idx = points[3]['index']
            duration = end_idx - start_idx
            
            # 检查持续时间
            if duration < min_duration:
                continue
            
            # 检查波动率
            box_height = resistance - support
            box_center = (resistance + support) / 2
            volatility = box_height / box_center
            
            if volatility > max_volatility:
                continue
            
            # 计算箱体内的量比特征
            box_volumes = volumes[start_idx:end_idx+1]
            volume_ratio = self.calculate_volume_ratio(box_volumes)
            avg_volume_ratio = np.mean(volume_ratio)
            
            box = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'resistance': resistance,
                'support': support,
                'duration': duration,
                'volatility': volatility,
                'strength': duration / 5 + (1 / volatility) * 2,  # 简单强度计算
                'avg_volume_ratio': avg_volume_ratio,
                'swing_points': points,
                'source': 'simple'
            }
            
            boxes.append(box)
        
        return self._filter_overlapping_boxes(boxes)
    
    def refine_box_boundaries_with_zigzag(self, box: dict, swing_points: List[dict], 
                                         highs: np.array, lows: np.array) -> dict:
        """
        使用ZigZag点重新确定箱体边界
        确保边界清晰且符合传统技术分析标准
        """
        start_idx = box['start_idx']
        end_idx = box['end_idx']
        
        # 找到箱体时间范围内的ZigZag点
        relevant_points = [p for p in swing_points 
                          if start_idx <= p['index'] <= end_idx]
        
        if len(relevant_points) >= 2:
            # 基于ZigZag点确定边界
            high_points = [p for p in relevant_points if p['type'] == 'high']
            low_points = [p for p in relevant_points if p['type'] == 'low']
            
            if high_points and low_points:
                # 使用ZigZag高低点作为边界
                resistance = max(p['price'] for p in high_points)
                support = min(p['price'] for p in low_points)
                
                # 更新箱体边界
                box['resistance'] = resistance
                box['support'] = support
                box['boundary_source'] = 'zigzag'
            else:
                box['boundary_source'] = 'original'
        else:
            box['boundary_source'] = 'original'
        
        return box
    
    def identify_boxes_hybrid(self, prices: np.array, volumes: np.array, 
                             highs: np.array, lows: np.array, macd_data: dict):
        """
        混合箱体识别算法
        实现用户要求的四个判断标准
        """
        print("🔄 执行混合箱体识别算法...")
        
        # 获取ZigZag摆动点
        swing_points, zigzag_line = self.zigzag_algorithm(highs, lows, deviation_pct=3.5)
        print(f"🔍 ZigZag识别到 {len(swing_points)} 个转折点")
        
        # 1. 第一判断标准：Enhanced逻辑识别箱体
        enhanced_boxes = self.identify_boxes_enhanced(prices, volumes, macd_data)
        print(f"📊 Enhanced逻辑识别到 {len(enhanced_boxes)} 个箱体")
        
        # 2. 箱体边界规则：基于ZigZag点确定边界
        for box in enhanced_boxes:
            box = self.refine_box_boundaries_with_zigzag(box, swing_points, highs, lows)
        
        # 3. 第二判断标准：Simple逻辑补充
        simple_boxes = self.identify_boxes_simple(prices, volumes, highs, lows)
        print(f"🔍 Simple逻辑识别到 {len(simple_boxes)} 个箱体")
        
        # 4. 冲突处理：合并结果，Enhanced优先
        final_boxes = self._merge_boxes_with_priority(enhanced_boxes, simple_boxes, swing_points, highs, lows)
        
        print(f"✅ 最终识别到 {len(final_boxes)} 个箱体")
        return final_boxes, swing_points, zigzag_line
    
    def _merge_boxes_with_priority(self, enhanced_boxes: List[dict], simple_boxes: List[dict],
                                  swing_points: List[dict], highs: np.array, lows: np.array) -> List[dict]:
        """
        合并Enhanced和Simple识别的箱体，Enhanced优先
        重叠箱体聚合为一个，使用Simple边界
        """
        all_boxes = []
        
        # 标记箱体来源并添加到统一列表
        for box in enhanced_boxes:
            box['source'] = 'enhanced'
            all_boxes.append(box)
        
        for box in simple_boxes:
            box['source'] = 'simple'
            all_boxes.append(box)
        
        # 按开始时间排序
        all_boxes.sort(key=lambda x: x['start_idx'])
        
        if not all_boxes:
            return []
        
        # 检测和聚合重叠箱体
        merged_boxes = []
        current_group = [all_boxes[0]]
        
        for i in range(1, len(all_boxes)):
            current_box = all_boxes[i]
            
            # 检查是否与当前组中的任何箱体重叠
            has_overlap = False
            for group_box in current_group:
                if self._boxes_overlap(current_box, group_box):
                    has_overlap = True
                    break
            
            if has_overlap:
                # 添加到当前重叠组
                current_group.append(current_box)
            else:
                # 处理当前组并开始新组
                merged_box = self._merge_overlapping_group(current_group, swing_points, highs, lows)
                merged_boxes.append(merged_box)
                current_group = [current_box]
        
        # 处理最后一组
        if current_group:
            merged_box = self._merge_overlapping_group(current_group, swing_points, highs, lows)
            merged_boxes.append(merged_box)
        
        print(f"📦 箱体聚合完成：{len(all_boxes)} -> {len(merged_boxes)} 个箱体")
        return merged_boxes
    
    def _boxes_overlap(self, box1: dict, box2: dict) -> bool:
        """
        检查两个箱体是否重叠（时间或价格空间）
        """
        # 时间重叠检查
        time_overlap = (box1['start_idx'] <= box2['end_idx'] and 
                       box1['end_idx'] >= box2['start_idx'])
        
        if not time_overlap:
            return False
        
        # 价格空间重叠检查
        price_overlap = (box1['support'] <= box2['resistance'] and 
                        box1['resistance'] >= box2['support'])
        
        return price_overlap
    
    def _merge_overlapping_group(self, box_group: List[dict], swing_points: List[dict], 
                                highs: np.array, lows: np.array) -> dict:
        """
        将重叠的箱体组聚合为一个箱体
        优先使用Enhanced逻辑，但边界采用Simple的ZigZag点
        """
        if len(box_group) == 1:
            # 单个箱体，确保使用ZigZag边界
            box = box_group[0]
            return self.refine_box_boundaries_with_zigzag(box, swing_points, highs, lows)
        
        # 多个重叠箱体，需要聚合
        print(f"🔄 聚合 {len(box_group)} 个重叠箱体")
        
        # 优先选择Enhanced箱体作为基础
        enhanced_boxes = [box for box in box_group if box['source'] == 'enhanced']
        simple_boxes = [box for box in box_group if box['source'] == 'simple']
        
        if enhanced_boxes:
            # 使用最强的Enhanced箱体作为基础
            base_box = max(enhanced_boxes, key=lambda x: x.get('strength', 0))
            merged_source = 'enhanced_merged'
        else:
            # 全部是Simple箱体，使用最强的
            base_box = max(simple_boxes, key=lambda x: x.get('strength', 0))
            merged_source = 'simple_merged'
        
        # 计算聚合后的时间范围
        start_idx = min(box['start_idx'] for box in box_group)
        end_idx = max(box['end_idx'] for box in box_group)
        
        # 创建聚合箱体
        merged_box = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'duration': end_idx - start_idx,
            'source': merged_source,
            'merged_count': len(box_group),
            'original_sources': [box['source'] for box in box_group],
            'strength': base_box.get('strength', 0) + len(box_group) * 0.5  # 聚合奖励
        }
        
        # 使用ZigZag点确定最终边界（Simple规则）
        merged_box = self.refine_box_boundaries_with_zigzag(merged_box, swing_points, highs, lows)
        
        # 如果ZigZag边界确定失败，使用组内边界的合理范围
        if merged_box.get('boundary_source') == 'original':
            all_resistances = [box['resistance'] for box in box_group]
            all_supports = [box['support'] for box in box_group]
            
            # 使用更保守的边界（更宽的范围）
            merged_box['resistance'] = max(all_resistances)
            merged_box['support'] = min(all_supports)
            merged_box['boundary_source'] = 'group_consensus'
        
        # 重新计算波动率
        if 'resistance' in merged_box and 'support' in merged_box:
            box_height = merged_box['resistance'] - merged_box['support']
            box_center = (merged_box['resistance'] + merged_box['support']) / 2
            merged_box['volatility'] = box_height / box_center
        
        return merged_box

    def _filter_overlapping_boxes(self, boxes):
        """过滤重叠的箱体，保留质量更好的"""
        if len(boxes) <= 1:
            return boxes
        
        # 按开始时间排序
        boxes.sort(key=lambda x: x['start_idx'])
        
        filtered = [boxes[0]]
        
        for current in boxes[1:]:
            last = filtered[-1]
            
            # 检查是否重叠
            if current['start_idx'] <= last['end_idx']:
                # 保留强度更高的箱体
                if current.get('strength', 0) > last.get('strength', 0):
                    filtered[-1] = current
            else:
                filtered.append(current)
        
        return filtered
    
    def detect_breakout_signals(self, boxes: List[dict], prices: np.array,
                               volumes: np.array, macd_data: dict) -> List[dict]:
        """检测突破信号"""
        signals = []
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        for box in boxes:
            end_idx = box['end_idx']
            resistance = box['resistance']
            support = box['support']
            
            # 检查箱体后的价格走势
            for i in range(end_idx + 1, min(end_idx + 30, len(prices))):
                current_price = prices[i]
                
                # 向上突破
                if current_price > resistance:
                    breakout_pct = (current_price - resistance) / resistance * 100
                    
                    # MACD确认
                    macd_confirmed = macd_data['macd'][i] > 0 and macd_data['dif'][i] > macd_data['dea'][i]
                    
                    # 成交量确认
                    volume_confirmed = volume_ratio[i] > 1.5
                    
                    # 计算信号强度
                    signal_strength = breakout_pct
                    if macd_confirmed:
                        signal_strength += 2
                    if volume_confirmed:
                        signal_strength += 3
                    
                    # 计算斐波那契目标位
                    fib_levels = self.calculate_fibonacci_levels(resistance, support)
                    targets = list(fib_levels['extension'].values())
                    
                    signal = {
                        'type': 'upward_breakout',
                        'breakout_idx': i,
                        'breakout_price': current_price,
                        'breakout_percentage': breakout_pct,
                        'macd_confirmed': macd_confirmed,
                        'volume_confirmed': volume_confirmed,
                        'volume_ratio': volume_ratio[i],
                        'signal_strength': signal_strength,
                        'fibonacci_targets': targets,
                        'box_info': box
                    }
                    signals.append(signal)
                    break
                
                # 向下突破
                elif current_price < support:
                    breakout_pct = (support - current_price) / support * 100
                    
                    # MACD确认
                    macd_confirmed = macd_data['macd'][i] < 0 and macd_data['dif'][i] < macd_data['dea'][i]
                    
                    # 成交量确认
                    volume_confirmed = volume_ratio[i] > 1.5
                    
                    # 计算信号强度
                    signal_strength = breakout_pct
                    if macd_confirmed:
                        signal_strength += 2
                    if volume_confirmed:
                        signal_strength += 3
                    
                    # 计算斐波那契目标位
                    fib_levels = self.calculate_fibonacci_levels(resistance, support)
                    targets = list(fib_levels['retracement'].values())
                    
                    signal = {
                        'type': 'downward_breakout',
                        'breakout_idx': i,
                        'breakout_price': current_price,
                        'breakout_percentage': breakout_pct,
                        'macd_confirmed': macd_confirmed,
                        'volume_confirmed': volume_confirmed,
                        'volume_ratio': volume_ratio[i],
                        'signal_strength': signal_strength,
                        'fibonacci_targets': targets,
                        'box_info': box
                    }
                    signals.append(signal)
                    break
        
        return signals
    
    def plot_advanced_analysis(self, data: pd.DataFrame, prices: np.array, volumes: np.array,
                              highs: np.array, lows: np.array, macd_data: dict):
        """绘制进阶版分析图表"""
        # 执行混合箱体识别
        boxes, swing_points, zigzag_line = self.identify_boxes_hybrid(prices, volumes, highs, lows, macd_data)
        
        # 检测突破信号
        breakout_signals = self.detect_breakout_signals(boxes, prices, volumes, macd_data)
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                           gridspec_kw={'height_ratios': [3, 1, 1]})
        
        dates = pd.to_datetime(data['date'])
        
        # 主图：K线和箱体
        self._plot_kline_with_boxes(ax1, data, dates, prices, boxes, breakout_signals, zigzag_line)
        
        # MACD图
        self._plot_macd(ax2, dates, macd_data)
        
        # 成交量图
        volume_ratio = self.calculate_volume_ratio(volumes)
        self._plot_volume(ax3, dates, volumes, volume_ratio)
        
        plt.tight_layout()
        
        # 保存图表
        if not os.path.exists('charts'):
            os.makedirs('charts')
        
        # 使用股票代码和时间戳生成唯一文件名
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = f'charts/advanced_box_breakout_{self.stock_code.replace(".", "_")}_{timestamp}'
        plt.savefig(f'{chart_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{chart_path}.jpg', dpi=300, bbox_inches='tight')
        
        print(f"进阶版分析图表已保存到charts目录: {os.path.basename(chart_path)}.png/.jpg")
        
        # 打印分析结果
        self._print_analysis_results(boxes, breakout_signals)
        
        return {
            'boxes': boxes,
            'signals': breakout_signals,
            'swing_points': swing_points,
            'zigzag_line': zigzag_line
        }
    
    def _plot_kline_with_boxes(self, ax, data, dates, prices, boxes, breakout_signals, zigzag_line):
        """绘制K线图和箱体"""
        # K线图
        for i in range(len(data)):
            date = dates[i]
            open_price = data.iloc[i]['open']
            high = data.iloc[i]['high']
            low = data.iloc[i]['low']
            close = prices[i]
            
            color = 'red' if close >= open_price else 'green'
            ax.plot([date, date], [low, high], color='black', linewidth=0.5)
            ax.plot([date, date], [open_price, close], color=color, linewidth=2)
        
        # ZigZag线
        if zigzag_line:
            zz_dates = [dates[point[0]] for point in zigzag_line]
            zz_prices = [point[1] for point in zigzag_line]
            ax.plot(zz_dates, zz_prices, 'purple', linewidth=2, alpha=0.7, label='ZigZag线')
        
        # 箱体和支撑阻力线
        for i, box in enumerate(boxes):
            start_date = dates[box['start_idx']]
            end_date = dates[min(box['end_idx'], len(dates)-1)]
            
            width_timedelta = end_date - start_date
            height = box['resistance'] - box['support']
            
            # 根据来源设置颜色
            if box.get('source') == 'enhanced':
                edge_color = 'orange'
                face_color = 'yellow'
                alpha = 0.3
            else:
                edge_color = 'blue'
                face_color = 'lightblue'
                alpha = 0.2
            
            # 绘制箱体矩形
            rect = plt.Rectangle((start_date, box['support']), 
                               width_timedelta, height,
                               linewidth=2, edgecolor=edge_color, facecolor=face_color, alpha=alpha)
            ax.add_patch(rect)
            
            # 绘制关键区域的支撑阻力线（延伸到箱体后一段时间）
            extension_days = min(30, len(dates) - box['end_idx'] - 1)
            if extension_days > 0:
                extended_end_idx = min(box['end_idx'] + extension_days, len(dates) - 1)
                extended_end_date = dates[extended_end_idx]
                
                # 支撑线
                ax.plot([start_date, extended_end_date], 
                       [box['support'], box['support']], 
                       color='blue', linestyle='--', linewidth=1.5, alpha=0.8, 
                       label='支撑线' if i == 0 else "")
                
                # 阻力线
                ax.plot([start_date, extended_end_date], 
                       [box['resistance'], box['resistance']], 
                       color='red', linestyle='--', linewidth=1.5, alpha=0.8, 
                       label='阻力线' if i == 0 else "")
            
            # 箱体标注
            mid_price = (box['resistance'] + box['support']) / 2
            mid_time = start_date + width_timedelta / 2
            
            # 根据来源显示不同信息
            if box.get('source') == 'enhanced':
                label_text = f'Box{i+1}(E)\n强度:{box["strength"]:.1f}\nMACD:{box.get("macd_score", 0):.1f}'
            else:
                label_text = f'Box{i+1}(S)\n强度:{box["strength"]:.1f}\n时长:{box["duration"]}天'
            
            ax.text(mid_time, mid_price, label_text,
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=face_color, alpha=0.7))
        
        # 绘制突破信号
        for signal in breakout_signals:
            idx = signal['breakout_idx']
            price = signal['breakout_price']
            
            if signal['type'] == 'upward_breakout':
                marker = '^'
                color = 'red'
                label = f"向上突破\n强度:{signal['signal_strength']:.1f}"
            else:
                marker = 'v'
                color = 'green'
                label = f"向下突破\n强度:{signal['signal_strength']:.1f}"
            
            ax.scatter(dates[idx], price, color=color, s=100, marker=marker, zorder=5)
            ax.annotate(label, (dates[idx], price), xytext=(10, 10), 
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax.set_title(f'{self.stock_code} 进阶版箱体突破分析 (Enhanced + Simple)', fontsize=14, fontweight='bold')
        ax.set_ylabel('价格', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_macd(self, ax, dates, macd_data):
        """绘制MACD指标"""
        ax.plot(dates, macd_data['dif'], label='DIF', color='blue', linewidth=1)
        ax.plot(dates, macd_data['dea'], label='DEA', color='red', linewidth=1)
        
        # MACD柱状图
        colors = ['red' if x > 0 else 'green' for x in macd_data['macd']]
        ax.bar(dates, macd_data['macd'], color=colors, alpha=0.6, width=1)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('MACD指标', fontsize=12)
        ax.set_ylabel('MACD', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_volume(self, ax, dates, volumes, volume_ratio):
        """绘制成交量和量比"""
        ax.bar(dates, volumes, alpha=0.6, color='blue', label='成交量')
        
        # 添加量比线
        ax2 = ax.twinx()
        ax2.plot(dates, volume_ratio, color='red', linewidth=1, label='量比')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        ax.set_title('成交量与量比', fontsize=12)
        ax.set_ylabel('成交量', fontsize=10)
        ax2.set_ylabel('量比', fontsize=10)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _print_analysis_results(self, boxes, breakout_signals):
        """打印分析结果"""
        print("\n" + "="*50)
        print(f"  {self.stock_code} 进阶版箱体突破分析结果")
        print("="*50)
        
        print(f"\n📊 箱体识别结果:")
        enhanced_count = sum(1 for box in boxes if box.get('source') == 'enhanced')
        simple_count = sum(1 for box in boxes if box.get('source') == 'simple')
        print(f"   总箱体数: {len(boxes)} 个")
        print(f"   Enhanced识别: {enhanced_count} 个")
        print(f"   Simple补充: {simple_count} 个")
        
        for i, box in enumerate(boxes):
            source_label = "Enhanced" if box.get('source') == 'enhanced' else "Simple"
            boundary_label = "ZigZag边界" if box.get('boundary_source') == 'zigzag' else "原始边界"
            
            print(f"\n箱体{i+1} ({source_label}): 支撑{box['support']:.2f} - 阻力{box['resistance']:.2f}")
            print(f"  综合强度: {box['strength']:.2f}")
            print(f"  波动率: {box['volatility']:.3f}, 持续时间: {box['duration']}天")
            print(f"  边界来源: {boundary_label}")
            
            if box.get('macd_score') is not None:
                print(f"  MACD稳定性: {box.get('macd_score', 0):.2f}")
        
        print(f"\n🚀 突破信号详情:")
        for i, signal in enumerate(breakout_signals):
            direction = "向上" if signal['type'] == 'upward_breakout' else "向下"
            print(f"\n信号{i+1}: {direction}突破")
            print(f"  突破价格: {signal['breakout_price']:.2f}, 突破幅度: {signal['breakout_percentage']:.2f}%")
            print(f"  MACD确认: {'是' if signal['macd_confirmed'] else '否'}")
            print(f"  成交量确认: {'是' if signal['volume_confirmed'] else '否'} (量比:{signal['volume_ratio']:.2f})")
            print(f"  信号强度: {signal['signal_strength']:.2f}")
            print(f"  斐波那契目标位: {[f'{x:.2f}' for x in signal['fibonacci_targets']]}")


def fetch_stock_data(stock_code: str, start_date: str = "2023-08-01", end_date: str = "2025-09-30"):
    """获取股票数据"""
    import baostock as bs
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return None
        
    rs = bs.query_history_k_data_plus(
        "sz.300992",
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
    print("🚀 启动进阶版箱体突破分析器 V3.1")
    print("=" * 50)
    
    # 获取数据
    data = fetch_stock_data("sz.300992", start_date="2023-08-01", end_date="2025-09-30")
    
    if data is None or len(data) == 0:
        print("❌ 无法获取股票数据")
        return
    
    print(f"成功获取{len(data)}条数据")
    
    # 创建分析器
    analyzer = AdvancedBoxBreakoutAnalyzer("sz.300992")
    
    # 数据预处理
    prices, volumes, highs, lows = analyzer.preprocess_data(data)
    
    # 计算技术指标
    macd_data = analyzer.calculate_macd(prices)
    
    # 执行分析并绘图
    results = analyzer.plot_advanced_analysis(data, prices, volumes, highs, lows, macd_data)
    
    # 保存结果（简化版本，避免JSON序列化问题）
    print("进阶版分析结果保存中...")
    
    # 只保存基本统计信息
    basic_results = {
        'stock_code': 'sz.300992',
        'analysis_date': datetime.now().isoformat(),
        'boxes_count': len(results['boxes']),
        'signals_count': len(results['signals']),
        'enhanced_boxes': sum(1 for box in results['boxes'] if box.get('source') == 'enhanced'),
        'simple_boxes': sum(1 for box in results['boxes'] if box.get('source') == 'simple')
    }
    
    with open('advanced_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(basic_results, f, ensure_ascii=False, indent=2)
    
    print("进阶版分析结果已保存到 advanced_analysis_results.json")


if __name__ == "__main__":
    main()