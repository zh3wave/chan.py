#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版箱体突破分析器 V2.0
移除MACD依赖，专注于价格突破和量比特征
集成分钟级量比分析功能
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
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SimplifiedBoxBreakoutAnalyzer:
    """
    简化版箱体突破分析器 v2.0
    专注于价格突破 + 量比特征，去掉MACD依赖
    """
    
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data = None
        self.minute_data = None  # 新增：分钟级数据
        self.volume_features = None  # 新增：量比特征
        
    def fetch_data(self, start_date: str = "2023-08-01", end_date: str = "2025-09-30"):
        """获取股票数据"""
        lg = bs.login()
        if lg.error_code != '0':
            print(f"登录失败: {lg.error_msg}")
            return None
            
        print(f"获取 {self.stock_code} 数据...")
        
        rs = bs.query_history_k_data_plus(
            self.stock_code,
            "date,open,high,low,close,volume,amount,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            print(f"查询失败: {rs.error_msg}")
            bs.logout()
            return None
            
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
            
        bs.logout()
        
        if not data_list:
            print("未获取到数据")
            return None
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        
        self.data = df
        print(f"成功获取 {len(df)} 条数据")
        return df
        
    def fetch_minute_data(self, start_date: str, end_date: str, frequency: str = "5"):
        """
        获取分钟级K线数据用于量比分析
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 分钟级别 "5", "15", "30", "60"
        """
        lg = bs.login()
        if lg.error_code != '0':
            print(f"登录失败: {lg.error_msg}")
            return None
            
        print(f"获取 {self.stock_code} {frequency}分钟K线数据...")
        
        rs = bs.query_history_k_data_plus(
            self.stock_code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            print(f"查询失败: {rs.error_msg}")
            bs.logout()
            return None
            
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
            
        bs.logout()
        
        if not data_list:
            print("未获取到分钟数据")
            return None
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 数据类型转换
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 处理baostock特殊的时间格式
        def parse_baostock_time(date_str, time_str):
            try:
                if len(time_str) >= 14:  # 格式: 20241101093500000
                    date_part = time_str[:8]  # 20241101
                    time_part = time_str[8:14]  # 093500
                    
                    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    
                    return pd.to_datetime(f"{formatted_date} {formatted_time}")
                else:
                    return pd.to_datetime(f"{date_str} {time_str}")
            except:
                return pd.NaT
        
        # 创建完整的datetime列
        df['datetime'] = df.apply(lambda row: parse_baostock_time(row['date'], row['time']), axis=1)
        df['date'] = pd.to_datetime(df['date'])
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S')
        
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        
        self.minute_data = df
        print(f"成功获取 {len(df)} 条分钟数据")
        return df
    
    def calculate_enhanced_volume_features(self) -> Dict:
        """
        计算增强的量比特征
        结合日线和分钟级数据
        
        Returns:
            包含各种量比特征的字典
        """
        if self.data is None:
            print("请先获取日线数据")
            return {}
            
        df = self.data.copy()
        
        # 基础量比计算
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # 开盘量比（基于分钟数据）
        opening_ratios = []
        intraday_patterns = []
        
        if self.minute_data is not None:
            minute_df = self.minute_data.copy()
            daily_df = df.copy()
            
            # 按日期处理分钟数据
            for date, day_data in minute_df.groupby('date'):
                if len(day_data) < 6:  # 数据太少跳过
                    continue
                    
                # 获取对应日线数据
                daily_info = daily_df[daily_df['date'] == date]
                if daily_info.empty:
                    continue
                    
                daily_volume = daily_info['volume'].iloc[0]
                volume_ma20 = daily_info['volume_ma20'].iloc[0]
                
                if pd.isna(volume_ma20) or volume_ma20 == 0:
                    continue
                    
                # 计算开盘量比（前30分钟，约6根5分钟K线）
                opening_data = day_data.head(6)
                opening_volume = opening_data['volume'].sum()
                
                # 估算开盘量比
                opening_ratio = (opening_volume * 8) / volume_ma20  # 8 = 240分钟/30分钟
                
                opening_ratios.append({
                    'date': date,
                    'opening_ratio': opening_ratio,
                    'opening_volume': opening_volume,
                    'daily_volume': daily_volume
                })
                
                # 计算日内量比变化模式
                day_data = day_data.copy()
                day_data['cumulative_volume'] = day_data['volume'].cumsum()
                day_data['time_progress'] = range(len(day_data))
                day_data['expected_volume'] = (day_data['time_progress'] + 1) / len(day_data) * daily_volume
                day_data['intraday_ratio'] = day_data['cumulative_volume'] / day_data['expected_volume']
                
                # 识别放量时点
                day_data['volume_spike'] = day_data['volume'] > day_data['volume'].rolling(window=3).mean() * 2
                surge_times = day_data[day_data['volume_spike']]['time'].tolist()
                
                intraday_patterns.append({
                    'date': date,
                    'volume_ratios': day_data['intraday_ratio'].tolist(),
                    'surge_times': surge_times,
                    'surge_count': len(surge_times)
                })
        
        # 将开盘量比合并到日线数据
        if opening_ratios:
            opening_df = pd.DataFrame(opening_ratios)
            df = df.merge(opening_df[['date', 'opening_ratio']], on='date', how='left')
        else:
            df['opening_ratio'] = np.nan
            
        # 计算综合量比特征
        df['enhanced_volume_score'] = 0
        
        # 日线量比权重 (40%)
        df['enhanced_volume_score'] += (df['volume_ratio'].fillna(1) - 1) * 0.4
        
        # 开盘量比权重 (60%) - 更重要的早期信号
        df['enhanced_volume_score'] += (df['opening_ratio'].fillna(1) - 1) * 0.6
        
        # 保存特征数据
        self.volume_features = {
            'daily_data': df,
            'opening_ratios': opening_ratios,
            'intraday_patterns': intraday_patterns
        }
        
        return self.volume_features
    
    def calculate_volume_ratio(self, volumes: np.array, period: int = 5):
        """计算量比指标"""
        volume_ma = pd.Series(volumes).rolling(window=period).mean().values
        volume_ratio = np.divide(volumes.astype(float), volume_ma.astype(float), 
                               out=np.ones_like(volumes, dtype=float), where=volume_ma!=0)
        return volume_ratio
    
    def calculate_opening_volume_ratio(self, volumes: np.array, period: int = 20):
        """
        计算开盘量比特征
        开盘量比具有特殊意义，是重要的动态指标
        """
        opening_volume_ratio = []
        for i in range(len(volumes)):
            if i < period:
                opening_volume_ratio.append(1.0)
            else:
                # 计算前N日同时段平均成交量
                avg_volume = np.mean(volumes[i-period:i])
                if avg_volume > 0:
                    ratio = volumes[i] / avg_volume
                    opening_volume_ratio.append(ratio)
                else:
                    opening_volume_ratio.append(1.0)
        
        return np.array(opening_volume_ratio)
    
    def zigzag_algorithm(self, highs: np.array, lows: np.array, deviation_pct: float = 3.5):
        """
        ZigZag算法识别价格摆动点，使用最高价和最低价
        """
        swing_points = []
        if len(highs) < 3 or len(lows) < 3:
            return swing_points
        
        # 寻找局部极值点
        high_indices = argrelextrema(highs, np.greater, order=5)[0]
        low_indices = argrelextrema(lows, np.less, order=5)[0]
        
        # 合并并排序极值点
        all_extrema = []
        for idx in high_indices:
            all_extrema.append((idx, highs[idx], 'high'))
        for idx in low_indices:
            all_extrema.append((idx, lows[idx], 'low'))
        
        all_extrema.sort(key=lambda x: x[0])
        
        if not all_extrema:
            return swing_points
        
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
        
        return swing_points
    
    def identify_boxes(self, prices: np.array, volumes: np.array, highs: np.array, lows: np.array,
                      min_duration: int = 8, max_volatility: float = 0.20):
        """
        识别价格箱体，使用适中的参数设置
        """
        swing_points = self.zigzag_algorithm(highs, lows, deviation_pct=3.5)  # 使用适中的ZigZag敏感度
        boxes = []
        
        print(f"🔍 ZigZag识别到 {len(swing_points)} 个转折点")
        
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
                'avg_volume_ratio': avg_volume_ratio,
                'swing_points': points
            }
            
            boxes.append(box)
        
        # 过滤重叠的箱体
        filtered_boxes = self._filter_overlapping_boxes(boxes)
        return filtered_boxes
    
    def _filter_overlapping_boxes(self, boxes):
        """过滤重叠的箱体，保留质量更好的"""
        if len(boxes) <= 1:
            return boxes
        
        # 按开始时间排序
        boxes.sort(key=lambda x: x['start_idx'])
        
        filtered_boxes = [boxes[0]]
        
        for current_box in boxes[1:]:
            last_box = filtered_boxes[-1]
            
            # 检查是否重叠
            if current_box['start_idx'] <= last_box['end_idx']:
                # 选择质量更好的箱体（持续时间更长，波动率更小）
                current_quality = current_box['duration'] / (1 + current_box['volatility'])
                last_quality = last_box['duration'] / (1 + last_box['volatility'])
                
                if current_quality > last_quality:
                    filtered_boxes[-1] = current_box
            else:
                filtered_boxes.append(current_box)
        
        return filtered_boxes
    
    def detect_breakout_signals(self) -> List[Dict]:
        """
        检测箱体突破信号
        集成增强的量比特征
        
        Returns:
            突破信号列表
        """
        if self.data is None:
            print("请先获取数据")
            return []
            
        # 计算增强量比特征
        volume_features = self.calculate_enhanced_volume_features()
        if not volume_features:
            print("量比特征计算失败")
            return []
            
        df = volume_features['daily_data']
        
        # 识别箱体
        df_data = df.copy()
        prices = df_data['close'].values
        volumes = df_data['volume'].values
        highs = df_data['high'].values
        lows = df_data['low'].values
        boxes = self.identify_boxes(prices, volumes, highs, lows)
        if not boxes:
            print("未识别到箱体")
            return []
            
        signals = []
        
        for box in boxes:
            start_idx = box['start_idx']
            end_idx = box['end_idx']
            resistance = box['resistance']
            support = box['support']
            
            # 检查箱体后的突破
            for i in range(end_idx + 1, len(df)):
                current_price = df.iloc[i]['close']
                current_volume = df.iloc[i]['volume']
                volume_ratio = df.iloc[i]['volume_ratio']
                opening_ratio = df.iloc[i]['opening_ratio'] if not pd.isna(df.iloc[i]['opening_ratio']) else 1.0
                enhanced_score = df.iloc[i]['enhanced_volume_score']
                
                # 向上突破检测
                if current_price > resistance * 1.02:  # 2%突破阈值
                    # 量比确认
                    volume_confirmed = volume_ratio > 1.5  # 日线量比确认
                    opening_confirmed = opening_ratio > 2.0  # 开盘量比确认
                    enhanced_confirmed = enhanced_score > 0.5  # 综合量比确认
                    
                    # 计算信号强度
                    breakout_strength = (current_price - resistance) / resistance * 100
                    
                    # 综合确认逻辑
                    confirmations = sum([volume_confirmed, opening_confirmed, enhanced_confirmed])
                    
                    signal = {
                        'date': df.iloc[i]['date'],
                        'type': 'upward_breakout',
                        'price': current_price,
                        'resistance': resistance,
                        'support': support,
                        'breakout_strength': breakout_strength,
                        'volume_ratio': volume_ratio,
                        'opening_ratio': opening_ratio,
                        'enhanced_score': enhanced_score,
                        'volume_confirmed': volume_confirmed,
                        'opening_confirmed': opening_confirmed,
                        'enhanced_confirmed': enhanced_confirmed,
                        'total_confirmations': confirmations,
                        'signal_quality': 'strong' if confirmations >= 2 else 'weak'
                    }
                    
                    signals.append(signal)
                    break  # 每个箱体只记录第一个突破
                    
                # 向下突破检测
                elif current_price < support * 0.98:  # 2%突破阈值
                    # 向下突破通常伴随放量
                    volume_confirmed = volume_ratio > 1.3
                    opening_confirmed = opening_ratio > 1.5
                    enhanced_confirmed = enhanced_score > 0.3
                    
                    breakout_strength = (support - current_price) / support * 100
                    
                    confirmations = sum([volume_confirmed, opening_confirmed, enhanced_confirmed])
                    
                    signal = {
                        'date': df.iloc[i]['date'],
                        'type': 'downward_breakout',
                        'price': current_price,
                        'resistance': resistance,
                        'support': support,
                        'breakout_strength': breakout_strength,
                        'volume_ratio': volume_ratio,
                        'opening_ratio': opening_ratio,
                        'enhanced_score': enhanced_score,
                        'volume_confirmed': volume_confirmed,
                        'opening_confirmed': opening_confirmed,
                        'enhanced_confirmed': enhanced_confirmed,
                        'total_confirmations': confirmations,
                        'signal_quality': 'strong' if confirmations >= 2 else 'weak'
                    }
                    
                    signals.append(signal)
                    break
                    
        return signals
    
    def plot_analysis(self, figsize=(18, 12)):
        """
        绘制简化版分析图表
        """
        if self.data is None:
            print("请先获取数据")
            return
        
        dates = pd.to_datetime(self.data['date'])
        prices = self.data['close'].values
        highs = self.data['high'].values
        lows = self.data['low'].values
        volumes = self.data['volume'].values
        
        # 计算技术指标
        volume_ratio = self.calculate_volume_ratio(volumes)
        opening_volume_ratio = self.calculate_opening_volume_ratio(volumes)
        
        # 识别箱体
        boxes = self.identify_boxes(prices, volumes, highs, lows)
        breakout_signals = self.detect_breakout_signals()
        swing_points = self.zigzag_algorithm(highs, lows)
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 1, 1])
        fig.suptitle(f'{self.stock_code} 简化版箱体突破分析 (v2.0)', fontsize=16, fontweight='bold')
        
        # 主图：K线图 + 箱体 + 突破信号
        self._plot_kline_with_boxes(axes[0], dates, prices, highs, lows, boxes, breakout_signals, swing_points)
        
        # 成交量图
        self._plot_volume(axes[1], dates, volumes, volume_ratio)
        
        # 开盘量比图
        self._plot_opening_volume_ratio(axes[2], dates, opening_volume_ratio)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simplified_test_{self.stock_code}_{timestamp}"
        
        # 确保charts目录存在
        charts_dir = 'charts'
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        # 保存图表到charts目录
        png_path = os.path.join(charts_dir, f"{filename}.png")
        jpg_path = os.path.join(charts_dir, f"{filename}.jpg")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(jpg_path, dpi=300, bbox_inches='tight')
        print(f"   简化版分析图已保存到charts目录: {filename}.png/.jpg")
        plt.show()
        
        # 打印分析结果
        print(f"\n📊 分析结果:")
        print(f"识别箱体数量: {len(boxes)}")
        print(f"突破信号数量: {len(breakout_signals)}")
        print(f"关键转折点数量: {len(swing_points)}")
        
        return {
            'boxes': boxes,
            'breakout_signals': breakout_signals,
            'swing_points': swing_points,
            'volume_ratio': volume_ratio,
            'opening_volume_ratio': opening_volume_ratio
        }
    
    def _plot_kline_with_boxes(self, ax, dates, prices, highs, lows, boxes, breakout_signals, swing_points):
        """绘制K线图和箱体"""
        # 绘制价格线
        ax.plot(dates, prices, 'b-', linewidth=1, alpha=0.7, label='收盘价')
        ax.fill_between(dates, lows, highs, alpha=0.1, color='gray', label='价格区间')
        
        # 绘制ZigZag线
        if swing_points:
            swing_dates = [dates.iloc[point['index']] for point in swing_points]
            swing_prices = [point['price'] for point in swing_points]
            ax.plot(swing_dates, swing_prices, 'g--', linewidth=1, alpha=0.6, label='ZigZag线')
            
            # 标记摆动点
            for point in swing_points:
                color = 'red' if point['type'] == 'high' else 'green'
                ax.scatter(dates.iloc[point['index']], point['price'], 
                          color=color, s=30, alpha=0.8, zorder=5)
        
        # 绘制箱体
        for i, box in enumerate(boxes):
            start_date = dates.iloc[box['start_idx']]
            end_date = dates.iloc[box['end_idx']]
            
            # 箱体矩形
            ax.axhspan(box['support'], box['resistance'], 
                      xmin=(box['start_idx'])/len(dates), 
                      xmax=(box['end_idx'])/len(dates),
                      alpha=0.2, color='orange', label='箱体' if i == 0 else "")
            
            # 支撑阻力线（只在箱体区域及稍后延伸）
            extend_length = min(20, len(dates) - box['end_idx'] - 1)
            extend_end_idx = box['end_idx'] + extend_length
            
            if extend_end_idx < len(dates):
                extend_start_date = dates.iloc[box['start_idx']]
                extend_end_date = dates.iloc[extend_end_idx]
                
                ax.hlines(box['resistance'], extend_start_date, extend_end_date, 
                         colors='red', linestyles='--', alpha=0.7, linewidth=1)
                ax.hlines(box['support'], extend_start_date, extend_end_date, 
                         colors='green', linestyles='--', alpha=0.7, linewidth=1)
        
        # 绘制突破信号
        for signal in breakout_signals:
            # 根据日期找到对应的索引
            signal_date_str = signal['date']
            try:
                signal_idx = dates[dates == signal_date_str].index[0]
                signal_date = dates.iloc[signal_idx]
                signal_price = signal['price']
                
                if signal['type'] == 'upward_breakout':
                    color = 'red'
                    marker = '^'
                    label = '向上突破'
                else:
                    color = 'blue'
                    marker = 'v'
                    label = '向下突破'
                
                ax.scatter(signal_date, signal_price, color=color, marker=marker, 
                          s=100, alpha=0.8, zorder=10, 
                          label=label if signal == breakout_signals[0] else "")
                
                # 添加信号强度标注
                ax.annotate(f"{signal['signal_quality']}\n{signal['breakout_strength']:.1f}%", 
                           (signal_date, signal_price), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            except (IndexError, KeyError):
                continue
        
        ax.set_title('价格走势与箱体突破信号')
        ax.set_ylabel('价格')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_volume(self, ax, dates, volumes, volume_ratio):
        """绘制成交量图"""
        ax.bar(dates, volumes, alpha=0.6, color='lightblue', label='成交量')
        
        # 绘制量比线
        ax2 = ax.twinx()
        ax2.plot(dates, volume_ratio, 'r-', linewidth=1, alpha=0.8, label='量比')
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='量比1.5')
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='量比2.0')
        
        ax.set_title('成交量与量比')
        ax.set_ylabel('成交量')
        ax2.set_ylabel('量比')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_opening_volume_ratio(self, ax, dates, opening_volume_ratio):
        """绘制开盘量比图"""
        ax.plot(dates, opening_volume_ratio, 'purple', linewidth=1, alpha=0.8, label='开盘量比')
        ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='开盘量比2.0')
        ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='开盘量比3.0')
        
        # 标记显著放量日
        significant_days = opening_volume_ratio > 2.0
        if np.any(significant_days):
            ax.scatter(dates[significant_days], opening_volume_ratio[significant_days], 
                      color='red', s=30, alpha=0.8, zorder=5, label='显著放量日')
        
        ax.set_title('开盘量比特征')
        ax.set_ylabel('开盘量比')
        ax.set_xlabel('日期')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def print_analysis_report(self, signals: List[Dict]):
        """打印分析报告"""
        if not signals:
            print("未检测到突破信号")
            return
            
        print("\n" + "="*60)
        print(f"📊 {self.stock_code} 箱体突破分析报告 V2.0")
        print("="*60)
        
        # 统计信息
        total_signals = len(signals)
        upward_signals = [s for s in signals if s['type'] == 'upward_breakout']
        downward_signals = [s for s in signals if s['type'] == 'downward_breakout']
        strong_signals = [s for s in signals if s['signal_quality'] == 'strong']
        
        print(f"\n📈 基础统计:")
        print(f"   总信号数: {total_signals}")
        print(f"   向上突破: {len(upward_signals)} ({len(upward_signals)/total_signals*100:.1f}%)")
        print(f"   向下突破: {len(downward_signals)} ({len(downward_signals)/total_signals*100:.1f}%)")
        print(f"   强信号数: {len(strong_signals)} ({len(strong_signals)/total_signals*100:.1f}%)")
        
        # 量比特征统计
        volume_confirmed = sum(1 for s in signals if s['volume_confirmed'])
        opening_confirmed = sum(1 for s in signals if s['opening_confirmed'])
        enhanced_confirmed = sum(1 for s in signals if s['enhanced_confirmed'])
        
        print(f"\n🔥 量比确认统计:")
        print(f"   日线量比确认: {volume_confirmed}/{total_signals} ({volume_confirmed/total_signals*100:.1f}%)")
        print(f"   开盘量比确认: {opening_confirmed}/{total_signals} ({opening_confirmed/total_signals*100:.1f}%)")
        print(f"   综合量比确认: {enhanced_confirmed}/{total_signals} ({enhanced_confirmed/total_signals*100:.1f}%)")
        
        # 平均指标
        avg_volume_ratio = np.mean([s['volume_ratio'] for s in signals])
        avg_opening_ratio = np.mean([s['opening_ratio'] for s in signals])
        avg_enhanced_score = np.mean([s['enhanced_score'] for s in signals])
        avg_breakout_strength = np.mean([s['breakout_strength'] for s in signals])
        
        print(f"\n📊 平均指标:")
        print(f"   平均日线量比: {avg_volume_ratio:.2f}")
        print(f"   平均开盘量比: {avg_opening_ratio:.2f}")
        print(f"   平均综合得分: {avg_enhanced_score:.2f}")
        print(f"   平均突破强度: {avg_breakout_strength:.2f}%")
        
        # 详细信号列表（前10个）
        print(f"\n🎯 详细信号 (前10个):")
        sorted_signals = sorted(signals, key=lambda x: x['total_confirmations'], reverse=True)[:10]
        
        for i, signal in enumerate(sorted_signals, 1):
            date_str = signal['date'].strftime('%Y-%m-%d')
            direction = "📈" if signal['type'] == 'upward_breakout' else "📉"
            quality = "🔥" if signal['signal_quality'] == 'strong' else "⚡"
            
            print(f"   {i:2d}. {date_str} {direction} {quality} "
                  f"价格:{signal['price']:.2f} "
                  f"强度:{signal['breakout_strength']:.1f}% "
                  f"量比:{signal['volume_ratio']:.1f} "
                  f"开盘量比:{signal['opening_ratio']:.1f} "
                  f"确认:{signal['total_confirmations']}/3")
        
        print("\n" + "="*60)

def main():
    """主函数 - 演示简化版箱体突破分析"""
    # 分析参数 - 按照图表时间跨度设置
    stock_code = "sz.000063"
    start_date = "2023-01-01"  # 从2023年1月开始
    end_date = "2025-01-24"    # 到2025年1月结束
    
    print(f"🚀 开始分析 {stock_code} 的箱体突破策略...")
    print(f"📅 分析时间范围: {start_date} 至 {end_date}")
    
    # 创建分析器
    analyzer = SimplifiedBoxBreakoutAnalyzer(stock_code)
    
    # 获取日线数据
    print("\n📊 获取日线数据...")
    daily_data = analyzer.fetch_data(start_date, end_date)
    if daily_data is None:
        print("❌ 日线数据获取失败")
        return
    
    # 获取分钟数据（用于量比分析）
    print("\n📊 获取分钟数据...")
    minute_data = analyzer.fetch_minute_data(start_date, end_date, frequency="5")
    if minute_data is None:
        print("⚠️ 分钟数据获取失败，将使用简化量比分析")
    
    # 检测突破信号
    print("\n🔍 检测箱体突破信号...")
    signals = analyzer.detect_breakout_signals()
    
    if not signals:
        print("❌ 未检测到突破信号")
        print("📊 生成基础分析图表...")
        analyzer.plot_analysis()
        return analyzer, []
    
    # 打印分析报告
    analyzer.print_analysis_report(signals)
    
    # 绘制分析图表
    print("\n📊 生成分析图表...")
    analyzer.plot_analysis()
    
    print(f"\n✅ 简化版箱体突破分析完成!")
    
    return analyzer, signals


if __name__ == "__main__":
    analyzer, signals = main()