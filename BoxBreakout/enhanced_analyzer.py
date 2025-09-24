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
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedBoxBreakoutAnalyzer:
    """
    增强版箱体突破分析器
    集成MACD、成交量、斐波那契等多维度技术指标验证
    """
    
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data = None
        
    def fetch_data(self, start_date: str = "2023-08-01", end_date: str = "2025-09-30"):
        """获取股票数据"""
        lg = bs.login()
        if lg.error_code != '0':
            print(f"登录失败: {lg.error_msg}")
            return None
            
        rs = bs.query_history_k_data_plus(
            self.stock_code,
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
        self.data = df
        return df
    
    def calculate_macd(self, prices: np.array, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9):
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
        macd_histogram = 2 * (dif - dea)
        
        return {
            'dif': dif,
            'dea': dea,
            'macd': macd_histogram,
            'fast_ema': fast_ema,
            'slow_ema': slow_ema
        }
    
    def calculate_volume_ratio(self, volumes: np.array, period: int = 5):
        """计算量比指标"""
        volume_ma = pd.Series(volumes).rolling(window=period).mean().values
        volume_ratio = np.divide(volumes.astype(float), volume_ma.astype(float), 
                               out=np.ones_like(volumes, dtype=float), where=volume_ma!=0)
        return volume_ratio
    
    def calculate_fibonacci_levels(self, high_price: float, low_price: float):
        """计算斐波那契回调位和扩展位"""
        price_range = high_price - low_price
        
        # 回调位
        retracement_levels = {
            '23.6%': high_price - 0.236 * price_range,
            '38.2%': high_price - 0.382 * price_range,
            '50.0%': high_price - 0.500 * price_range,
            '61.8%': high_price - 0.618 * price_range,
            '78.6%': high_price - 0.786 * price_range
        }
        
        # 扩展位
        extension_levels = {
            '127.2%': high_price + 0.272 * price_range,
            '161.8%': high_price + 0.618 * price_range,
            '261.8%': high_price + 1.618 * price_range
        }
        
        return retracement_levels, extension_levels
    
    def zigzag_algorithm(self, prices: np.array, deviation_pct: float = 5.0):
        """ZigZag算法识别摆动点"""
        swing_points = []
        
        if len(prices) < 3:
            return swing_points, []
            
        # 初始化
        last_swing_idx = 0
        last_swing_price = prices[0]
        trend_direction = None
        zigzag_line = []
        
        for i in range(1, len(prices)):
            current_price = prices[i]
            price_change_pct = abs(current_price - last_swing_price) / last_swing_price * 100
            
            if price_change_pct >= deviation_pct:
                if current_price > last_swing_price:
                    if trend_direction == 'down':
                        swing_points.append({
                            'index': last_swing_idx,
                            'price': last_swing_price,
                            'type': 'low'
                        })
                        zigzag_line.append((last_swing_idx, last_swing_price))
                    trend_direction = 'up'
                else:
                    if trend_direction == 'up':
                        swing_points.append({
                            'index': last_swing_idx,
                            'price': last_swing_price,
                            'type': 'high'
                        })
                        zigzag_line.append((last_swing_idx, last_swing_price))
                    trend_direction = 'down'
                
                last_swing_idx = i
                last_swing_price = current_price
        
        # 添加最后一个点
        zigzag_line.append((last_swing_idx, last_swing_price))
        
        return swing_points, zigzag_line
    
    def identify_boxes_with_indicators(self, prices: np.array, volumes: np.array, 
                                     macd_data: dict, min_duration: int = 15, 
                                     max_volatility: float = 0.12):
        """
        增强版箱体识别算法
        结合MACD和成交量特征进行验证
        """
        boxes = []
        window_size = min_duration
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        for i in range(window_size, len(prices) - window_size, 5):
            window_prices = prices[i-window_size:i+window_size]
            window_volumes = volumes[i-window_size:i+window_size]
            window_macd = macd_data['macd'][i-window_size:i+window_size]
            window_vol_ratio = volume_ratio[i-window_size:i+window_size]
            
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
                        'end_idx': i + window_size,
                        'resistance': window_high,
                        'support': window_low,
                        'duration': window_size * 2,
                        'volatility': volatility,
                        'strength': total_strength,
                        'base_strength': base_strength,
                        'macd_score': macd_score,
                        'volume_score': volume_score,
                        'macd_volatility': macd_volatility,
                        'volume_stability': volume_stability
                    }
                    boxes.append(box)
        
        # 过滤重叠箱体
        filtered_boxes = self._filter_overlapping_boxes(boxes)
        return filtered_boxes
    
    def _filter_overlapping_boxes(self, boxes):
        """过滤重叠的箱体，保留强度最高的"""
        if not boxes:
            return []
        
        boxes.sort(key=lambda x: x['strength'], reverse=True)
        
        filtered = []
        for box in boxes:
            overlapping = False
            for existing_box in filtered:
                if (box['start_idx'] <= existing_box['end_idx'] and 
                    box['end_idx'] >= existing_box['start_idx']):
                    overlapping = True
                    break
            
            if not overlapping:
                filtered.append(box)
        
        return filtered[:8]
    
    def detect_breakout_signals(self, boxes: List[dict], prices: np.array, 
                              volumes: np.array, macd_data: dict):
        """
        检测箱体突破信号
        结合MACD和成交量进行双重验证
        """
        breakout_signals = []
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        for box in boxes:
            end_idx = min(box['end_idx'], len(prices) - 1)
            
            # 检查箱体后的价格走势
            for i in range(end_idx + 1, min(end_idx + 20, len(prices))):
                current_price = prices[i]
                current_volume_ratio = volume_ratio[i] if i < len(volume_ratio) else 1.0
                
                # 向上突破检测
                if current_price > box['resistance']:
                    # 价格突破验证
                    breakout_pct = (current_price - box['resistance']) / box['resistance'] * 100
                    
                    # MACD验证：DIF上穿DEA或MACD柱状图转正
                    macd_confirm = False
                    if i < len(macd_data['dif']) and i < len(macd_data['dea']):
                        if (macd_data['dif'][i] > macd_data['dea'][i] and 
                            macd_data['macd'][i] > 0):
                            macd_confirm = True
                    
                    # 成交量验证：量比放大
                    volume_confirm = current_volume_ratio > 1.5
                    
                    # 计算斐波那契目标位
                    box_height = box['resistance'] - box['support']
                    fib_targets = {
                        '127.2%': box['resistance'] + 0.272 * box_height,
                        '161.8%': box['resistance'] + 0.618 * box_height,
                        '261.8%': box['resistance'] + 1.618 * box_height
                    }
                    
                    signal = {
                        'type': 'upward_breakout',
                        'box_id': boxes.index(box),
                        'breakout_idx': i,
                        'breakout_price': current_price,
                        'breakout_pct': breakout_pct,
                        'resistance_level': box['resistance'],
                        'support_level': box['support'],
                        'macd_confirm': macd_confirm,
                        'volume_confirm': volume_confirm,
                        'volume_ratio': current_volume_ratio,
                        'signal_strength': breakout_pct + (2 if macd_confirm else 0) + (1 if volume_confirm else 0),
                        'fibonacci_targets': fib_targets
                    }
                    breakout_signals.append(signal)
                    break
                
                # 向下突破检测
                elif current_price < box['support']:
                    breakout_pct = (box['support'] - current_price) / box['support'] * 100
                    
                    # MACD验证：DIF下穿DEA或MACD柱状图转负
                    macd_confirm = False
                    if i < len(macd_data['dif']) and i < len(macd_data['dea']):
                        if (macd_data['dif'][i] < macd_data['dea'][i] and 
                            macd_data['macd'][i] < 0):
                            macd_confirm = True
                    
                    # 成交量验证
                    volume_confirm = current_volume_ratio > 1.5
                    
                    # 计算斐波那契目标位
                    box_height = box['resistance'] - box['support']
                    fib_targets = {
                        '127.2%': box['support'] - 0.272 * box_height,
                        '161.8%': box['support'] - 0.618 * box_height,
                        '261.8%': box['support'] - 1.618 * box_height
                    }
                    
                    signal = {
                        'type': 'downward_breakout',
                        'box_id': boxes.index(box),
                        'breakout_idx': i,
                        'breakout_price': current_price,
                        'breakout_pct': breakout_pct,
                        'resistance_level': box['resistance'],
                        'support_level': box['support'],
                        'macd_confirm': macd_confirm,
                        'volume_confirm': volume_confirm,
                        'volume_ratio': current_volume_ratio,
                        'signal_strength': breakout_pct + (2 if macd_confirm else 0) + (1 if volume_confirm else 0),
                        'fibonacci_targets': fib_targets
                    }
                    breakout_signals.append(signal)
                    break
        
        return breakout_signals
    
    def plot_enhanced_analysis(self, figsize=(20, 15)):
        """绘制增强版分析图表"""
        if self.data is None:
            print("请先获取数据")
            return
        
        prices = self.data['close'].values
        highs = self.data['high'].values  
        lows = self.data['low'].values
        volumes = self.data['volume'].values
        dates = pd.to_datetime(self.data['date']).dt.to_pydatetime()
        
        # 计算技术指标
        macd_data = self.calculate_macd(prices)
        volume_ratio = self.calculate_volume_ratio(volumes)
        
        # 执行分析
        swing_points, zigzag_line = self.zigzag_algorithm(prices)
        boxes = self.identify_boxes_with_indicators(prices, volumes, macd_data)
        breakout_signals = self.detect_breakout_signals(boxes, prices, volumes, macd_data)
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, 
                                                    height_ratios=[2, 1], width_ratios=[3, 1])
        
        # 主图：K线图 + 箱体 + 突破信号
        self._plot_kline_with_boxes(ax1, dates, prices, highs, lows, boxes, 
                                   breakout_signals, zigzag_line)
        
        # MACD图
        self._plot_macd(ax2, dates, macd_data)
        
        # 成交量图
        self._plot_volume(ax3, dates, volumes, volume_ratio)
        
        # 信号统计图
        self._plot_signal_statistics(ax4, breakout_signals, boxes)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('enhanced_box_breakout_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('enhanced_box_breakout_analysis.jpg', dpi=300, bbox_inches='tight')
        print("增强版分析图表已保存")
        
        plt.show()
        
        # 打印分析结果
        self._print_analysis_results(boxes, breakout_signals, swing_points)
        
        return {
            'boxes': boxes,
            'breakout_signals': breakout_signals,
            'swing_points': swing_points,
            'macd_data': macd_data,
            'volume_ratio': volume_ratio
        }
    
    def _plot_kline_with_boxes(self, ax, dates, prices, highs, lows, boxes, 
                              breakout_signals, zigzag_line):
        """绘制K线图和箱体"""
        # K线图
        for i in range(len(self.data)):
            date = dates[i]
            open_price = self.data.iloc[i]['open']
            high = highs[i]
            low = lows[i]
            close = prices[i]
            
            color = 'red' if close >= open_price else 'green'
            ax.plot([date, date], [low, high], color='black', linewidth=0.5)
            ax.plot([date, date], [open_price, close], color=color, linewidth=2)
        
        # ZigZag线
        if zigzag_line:
            zz_dates = [dates[point[0]] for point in zigzag_line]
            zz_prices = [point[1] for point in zigzag_line]
            ax.plot(zz_dates, zz_prices, 'purple', linewidth=2, alpha=0.7, label='ZigZag线')
        
        # 绘制箱体
        for i, box in enumerate(boxes):
            start_date = dates[box['start_idx']]
            end_date = dates[min(box['end_idx'], len(dates)-1)]
            
            width_timedelta = end_date - start_date
            height = box['resistance'] - box['support']
            
            rect = plt.Rectangle((start_date, box['support']), 
                               width_timedelta, height,
                               linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.2)
            ax.add_patch(rect)
            
            # 标注箱体信息
            mid_price = (box['resistance'] + box['support']) / 2
            mid_time = start_date + width_timedelta / 2
            ax.text(mid_time, mid_price, 
                    f'Box{i+1}\n强度:{box["strength"]:.1f}\nMACD:{box["macd_score"]:.1f}', 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
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
        
        ax.set_title(f'{self.stock_code} 增强版箱体突破分析', fontsize=14, fontweight='bold')
        ax.set_ylabel('价格', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_macd(self, ax, dates, macd_data):
        """绘制MACD指标"""
        ax.plot(dates, macd_data['dif'], label='DIF', color='blue', linewidth=1)
        ax.plot(dates, macd_data['dea'], label='DEA', color='red', linewidth=1)
        
        # MACD柱状图
        colors = ['red' if x > 0 else 'green' for x in macd_data['macd']]
        ax.bar(dates, macd_data['macd'], color=colors, alpha=0.6, width=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('MACD指标', fontsize=12)
        ax.set_ylabel('MACD', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_volume(self, ax, dates, volumes, volume_ratio):
        """绘制成交量和量比"""
        ax.bar(dates, volumes, color='gray', alpha=0.6, width=0.8, label='成交量')
        
        # 添加量比线
        ax2 = ax.twinx()
        ax2.plot(dates, volume_ratio, color='orange', linewidth=2, label='量比')
        ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='量比1.5')
        
        ax.set_title('成交量与量比', fontsize=12)
        ax.set_ylabel('成交量', fontsize=10)
        ax2.set_ylabel('量比', fontsize=10)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_signal_statistics(self, ax, breakout_signals, boxes):
        """绘制信号统计"""
        if not breakout_signals:
            ax.text(0.5, 0.5, '暂无突破信号', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('突破信号统计', fontsize=12)
            return
        
        # 统计信号类型
        upward_count = sum(1 for s in breakout_signals if s['type'] == 'upward_breakout')
        downward_count = len(breakout_signals) - upward_count
        
        # 统计验证情况
        macd_confirmed = sum(1 for s in breakout_signals if s['macd_confirm'])
        volume_confirmed = sum(1 for s in breakout_signals if s['volume_confirm'])
        double_confirmed = sum(1 for s in breakout_signals if s['macd_confirm'] and s['volume_confirm'])
        
        # 绘制饼图
        labels = ['向上突破', '向下突破']
        sizes = [upward_count, downward_count]
        colors = ['red', 'green']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'突破信号统计\n总计:{len(breakout_signals)}个\nMACD确认:{macd_confirmed}个\n成交量确认:{volume_confirmed}个\n双重确认:{double_confirmed}个', 
                    fontsize=10)
    
    def _print_analysis_results(self, boxes, breakout_signals, swing_points):
        """打印分析结果"""
        print(f"\n=== {self.stock_code} 增强版分析结果 ===")
        print(f"识别摆动点数量: {len(swing_points)}")
        print(f"识别箱体数量: {len(boxes)}")
        print(f"检测突破信号数量: {len(breakout_signals)}")
        
        print(f"\n=== 箱体详情 ===")
        for i, box in enumerate(boxes):
            print(f"箱体{i+1}: 支撑{box['support']:.2f} - 阻力{box['resistance']:.2f}")
            print(f"  综合强度: {box['strength']:.2f} (基础:{box['base_strength']:.0f} + MACD:{box['macd_score']:.1f} + 成交量:{box['volume_score']:.1f})")
            print(f"  波动率: {box['volatility']:.3f}, MACD稳定性: {box['macd_volatility']:.3f}")
        
        print(f"\n=== 突破信号详情 ===")
        for i, signal in enumerate(breakout_signals):
            print(f"信号{i+1}: {signal['type']}")
            print(f"  突破价格: {signal['breakout_price']:.2f}, 突破幅度: {signal['breakout_pct']:.2f}%")
            print(f"  MACD确认: {'是' if signal['macd_confirm'] else '否'}")
            print(f"  成交量确认: {'是' if signal['volume_confirm'] else '否'} (量比:{signal['volume_ratio']:.2f})")
            print(f"  信号强度: {signal['signal_strength']:.2f}")
            print(f"  斐波那契目标位: {list(signal['fibonacci_targets'].values())}")


def main():
    """主函数"""
    print("开始增强版箱体突破分析...")
    
    # 创建分析器
    analyzer = EnhancedBoxBreakoutAnalyzer("sz.000063")
    
    # 获取数据
    data = analyzer.fetch_data("2023-08-01", "2025-09-30")
    
    if data is not None:
        print(f"成功获取{len(data)}条数据")
        
        # 执行增强版分析
        results = analyzer.plot_enhanced_analysis()
        
        # 保存结果
        if results:
            import json
            with open('enhanced_analysis_results.json', 'w', encoding='utf-8') as f:
                # 处理numpy类型
                json_results = {}
                for key, value in results.items():
                    if key in ['macd_data', 'volume_ratio']:
                        if isinstance(value, dict):
                            json_results[key] = {k: v.tolist() if hasattr(v, 'tolist') else v 
                                               for k, v in value.items()}
                        else:
                            json_results[key] = value.tolist() if hasattr(value, 'tolist') else value
                    else:
                        json_results[key] = value
                        
                json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
            print("增强版分析结果已保存到 enhanced_analysis_results.json")
    else:
        print("数据获取失败！")

if __name__ == "__main__":
    main()