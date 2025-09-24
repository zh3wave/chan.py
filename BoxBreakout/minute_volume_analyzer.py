#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分钟级量比分析器
基于baostock分钟K线数据计算量比特征，作为分时量比的替代方案
"""

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class MinuteVolumeAnalyzer:
    """
    分钟级量比分析器
    基于分钟K线数据计算量比特征，模拟分时量比效果
    """
    
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.minute_data = None
        self.daily_data = None
        
    def fetch_minute_data(self, start_date: str, end_date: str, frequency: str = "5"):
        """
        获取分钟级K线数据
        
        Args:
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD" 
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
        # baostock分钟数据的time格式可能是 "20241101093500000"
        def parse_baostock_time(date_str, time_str):
            try:
                if len(time_str) >= 14:  # 格式: 20241101093500000
                    # 提取日期和时间部分
                    date_part = time_str[:8]  # 20241101
                    time_part = time_str[8:14]  # 093500
                    
                    # 格式化为标准时间
                    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    
                    return pd.to_datetime(f"{formatted_date} {formatted_time}")
                else:
                    # 如果是标准格式，直接解析
                    return pd.to_datetime(f"{date_str} {time_str}")
            except:
                return pd.NaT
        
        # 创建完整的datetime列
        df['datetime'] = df.apply(lambda row: parse_baostock_time(row['date'], row['time']), axis=1)
        df['date'] = pd.to_datetime(df['date'])
        
        # 从datetime中提取标准时间格式
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S')
        
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        
        self.minute_data = df
        print(f"成功获取 {len(df)} 条分钟数据")
        return df
        
    def fetch_daily_data(self, start_date: str, end_date: str):
        """获取日线数据用于计算基准量比"""
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
        
        if not data_list:
            return None
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()
        self.daily_data = df
        return df
        
    def calculate_volume_ratio_features(self) -> Dict:
        """
        计算基于分钟数据的量比特征
        
        Returns:
            包含各种量比特征的字典
        """
        if self.minute_data is None or self.daily_data is None:
            print("请先获取分钟和日线数据")
            return {}
            
        minute_df = self.minute_data.copy()
        daily_df = self.daily_data.copy()
        
        # 按日期分组计算量比特征
        features = {
            'opening_volume_ratio': [],  # 开盘量比
            'intraday_volume_patterns': [],  # 日内量比模式
            'volume_surge_times': [],  # 放量时点
            'daily_volume_distribution': []  # 日内成交量分布
        }
        
        # 计算历史平均成交量（用于量比基准）
        daily_df['volume_ma5'] = daily_df['volume'].rolling(window=5).mean()
        daily_df['volume_ma20'] = daily_df['volume'].rolling(window=20).mean()
        
        # 按日期处理分钟数据
        for date, day_data in minute_df.groupby('date'):
            if len(day_data) < 10:  # 数据太少跳过
                continue
                
            # 获取对应日线数据
            daily_info = daily_df[daily_df['date'] == date]
            if daily_info.empty:
                continue
                
            daily_volume = daily_info['volume'].iloc[0]
            volume_ma20 = daily_info['volume_ma20'].iloc[0]
            
            if pd.isna(volume_ma20) or volume_ma20 == 0:
                continue
                
            # 计算开盘量比（前30分钟）
            opening_data = day_data.head(6)  # 5分钟K线，6根约30分钟
            opening_volume = opening_data['volume'].sum()
            
            # 估算开盘量比（开盘30分钟成交量 vs 历史同期平均）
            opening_ratio = (opening_volume * 8) / volume_ma20  # 8 = 240分钟/30分钟
            
            # 计算日内量比变化
            day_data = day_data.copy()
            day_data['cumulative_volume'] = day_data['volume'].cumsum()
            day_data['time_progress'] = range(len(day_data))
            day_data['expected_volume'] = (day_data['time_progress'] + 1) / len(day_data) * daily_volume
            day_data['volume_ratio'] = day_data['cumulative_volume'] / day_data['expected_volume']
            
            # 识别放量时点（量比突然增大）
            day_data['volume_spike'] = day_data['volume'] > day_data['volume'].rolling(window=3).mean() * 2
            surge_times = day_data[day_data['volume_spike']]['time'].tolist()
            
            # 保存特征
            features['opening_volume_ratio'].append({
                'date': date,
                'opening_ratio': opening_ratio,
                'opening_volume': opening_volume,
                'daily_volume': daily_volume
            })
            
            features['intraday_volume_patterns'].append({
                'date': date,
                'volume_ratios': day_data['volume_ratio'].tolist(),
                'times': day_data['time'].tolist()
            })
            
            features['volume_surge_times'].append({
                'date': date,
                'surge_times': surge_times,
                'surge_count': len(surge_times)
            })
            
            # 计算成交量分布特征
            total_volume = day_data['volume'].sum()
            morning_volume = day_data[day_data['time'] <= '11:30:00']['volume'].sum()
            afternoon_volume = total_volume - morning_volume
            
            features['daily_volume_distribution'].append({
                'date': date,
                'morning_ratio': morning_volume / total_volume if total_volume > 0 else 0,
                'afternoon_ratio': afternoon_volume / total_volume if total_volume > 0 else 0,
                'total_volume': total_volume
            })
            
        return features
        
    def analyze_volume_patterns(self, features: Dict) -> Dict:
        """
        分析量比模式，识别异常放量
        
        Args:
            features: calculate_volume_ratio_features返回的特征字典
            
        Returns:
            分析结果字典
        """
        analysis = {
            'high_opening_ratio_days': [],  # 开盘高量比日期
            'intraday_surge_patterns': [],  # 日内放量模式
            'volume_distribution_analysis': {},  # 成交量分布分析
            'statistics': {}
        }
        
        # 分析开盘量比
        opening_ratios = [item['opening_ratio'] for item in features['opening_volume_ratio']]
        if opening_ratios:
            ratio_mean = np.mean(opening_ratios)
            ratio_std = np.std(opening_ratios)
            threshold = ratio_mean + 1.5 * ratio_std
            
            for item in features['opening_volume_ratio']:
                if item['opening_ratio'] > threshold:
                    analysis['high_opening_ratio_days'].append({
                        'date': item['date'],
                        'opening_ratio': item['opening_ratio'],
                        'significance': (item['opening_ratio'] - ratio_mean) / ratio_std
                    })
        
        # 分析日内放量模式
        for item in features['volume_surge_times']:
            if item['surge_count'] >= 3:  # 多次放量
                analysis['intraday_surge_patterns'].append({
                    'date': item['date'],
                    'surge_count': item['surge_count'],
                    'surge_times': item['surge_times']
                })
        
        # 统计信息
        analysis['statistics'] = {
            'total_days': len(features['opening_volume_ratio']),
            'high_opening_days': len(analysis['high_opening_ratio_days']),
            'surge_pattern_days': len(analysis['intraday_surge_patterns']),
            'avg_opening_ratio': np.mean(opening_ratios) if opening_ratios else 0
        }
        
        return analysis
        
    def plot_volume_analysis(self, features: Dict, analysis: Dict, save_path: str = None):
        """
        绘制量比分析图表
        
        Args:
            features: 量比特征数据
            analysis: 分析结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.stock_code} 分钟级量比特征分析', fontsize=16, fontweight='bold')
        
        # 1. 开盘量比时序图
        ax1 = axes[0, 0]
        opening_data = features['opening_volume_ratio']
        dates = [item['date'] for item in opening_data]
        ratios = [item['opening_ratio'] for item in opening_data]
        
        ax1.plot(dates, ratios, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=np.mean(ratios), color='r', linestyle='--', alpha=0.7, label=f'均值: {np.mean(ratios):.2f}')
        
        # 标记高量比日期
        for item in analysis['high_opening_ratio_days']:
            ax1.scatter(item['date'], item['opening_ratio'], color='red', s=50, zorder=5)
            
        ax1.set_title('开盘量比时序')
        ax1.set_ylabel('开盘量比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 开盘量比分布直方图
        ax2 = axes[0, 1]
        ax2.hist(ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=np.mean(ratios), color='r', linestyle='--', label=f'均值: {np.mean(ratios):.2f}')
        ax2.axvline(x=np.mean(ratios) + 1.5*np.std(ratios), color='orange', linestyle='--', 
                   label=f'异常阈值: {np.mean(ratios) + 1.5*np.std(ratios):.2f}')
        ax2.set_title('开盘量比分布')
        ax2.set_xlabel('开盘量比')
        ax2.set_ylabel('频次')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 日内放量次数统计
        ax3 = axes[1, 0]
        surge_counts = [item['surge_count'] for item in features['volume_surge_times']]
        surge_dates = [item['date'] for item in features['volume_surge_times']]
        
        colors = ['red' if count >= 3 else 'blue' for count in surge_counts]
        ax3.scatter(surge_dates, surge_counts, c=colors, alpha=0.6)
        ax3.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='多次放量阈值')
        ax3.set_title('日内放量次数')
        ax3.set_ylabel('放量次数')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计信息表格
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = analysis['statistics']
        stats_text = f"""
        分析统计信息:
        
        总分析天数: {stats['total_days']}
        高开盘量比天数: {stats['high_opening_days']}
        多次放量天数: {stats['surge_pattern_days']}
        平均开盘量比: {stats['avg_opening_ratio']:.2f}
        
        异常放量比例: {stats['high_opening_days']/stats['total_days']*100:.1f}%
        多次放量比例: {stats['surge_pattern_days']/stats['total_days']*100:.1f}%
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
        
    def print_analysis_report(self, analysis: Dict):
        """打印分析报告"""
        print("\n" + "="*60)
        print(f"📊 {self.stock_code} 分钟级量比特征分析报告")
        print("="*60)
        
        stats = analysis['statistics']
        print(f"\n📈 基础统计:")
        print(f"   总分析天数: {stats['total_days']}")
        print(f"   平均开盘量比: {stats['avg_opening_ratio']:.2f}")
        print(f"   高开盘量比天数: {stats['high_opening_days']} ({stats['high_opening_days']/stats['total_days']*100:.1f}%)")
        print(f"   多次放量天数: {stats['surge_pattern_days']} ({stats['surge_pattern_days']/stats['total_days']*100:.1f}%)")
        
        print(f"\n🔥 异常开盘量比日期 (前10个):")
        high_days = sorted(analysis['high_opening_ratio_days'], 
                          key=lambda x: x['opening_ratio'], reverse=True)[:10]
        for item in high_days:
            print(f"   {item['date'].strftime('%Y-%m-%d')}: 量比 {item['opening_ratio']:.2f} "
                  f"(显著性 {item['significance']:.1f}σ)")
        
        print(f"\n⚡ 日内多次放量模式 (前5个):")
        surge_days = sorted(analysis['intraday_surge_patterns'], 
                           key=lambda x: x['surge_count'], reverse=True)[:5]
        for item in surge_days:
            print(f"   {item['date'].strftime('%Y-%m-%d')}: {item['surge_count']}次放量 "
                  f"时点: {', '.join(item['surge_times'][:3])}")
        
        print("\n" + "="*60)


def main():
    """主函数 - 演示分钟级量比分析"""
    # 分析参数
    stock_code = "sz.000063"
    start_date = "2024-11-01"
    end_date = "2025-01-24"
    
    print(f"🚀 开始分析 {stock_code} 的分钟级量比特征...")
    
    # 创建分析器
    analyzer = MinuteVolumeAnalyzer(stock_code)
    
    # 获取数据
    print("\n📊 获取分钟和日线数据...")
    minute_data = analyzer.fetch_minute_data(start_date, end_date, frequency="5")
    daily_data = analyzer.fetch_daily_data(start_date, end_date)
    
    if minute_data is None or daily_data is None:
        print("❌ 数据获取失败")
        return
    
    # 计算量比特征
    print("\n🔍 计算量比特征...")
    features = analyzer.calculate_volume_ratio_features()
    
    if not features['opening_volume_ratio']:
        print("❌ 量比特征计算失败")
        return
    
    # 分析量比模式
    print("\n📈 分析量比模式...")
    analysis = analyzer.analyze_volume_patterns(features)
    
    # 打印报告
    analyzer.print_analysis_report(analysis)
    
    # 绘制图表
    print("\n📊 生成分析图表...")
    save_path = f"minute_volume_analysis_{stock_code.replace('.', '_')}.png"
    analyzer.plot_volume_analysis(features, analysis, save_path)
    
    print(f"\n✅ 分钟级量比分析完成!")
    
    return analyzer, features, analysis


if __name__ == "__main__":
    analyzer, features, analysis = main()