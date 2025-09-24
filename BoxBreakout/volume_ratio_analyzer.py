import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import baostock as bs
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VolumeRatioAnalyzer:
    """量比特征分析器 - 专注于开盘量比和日内量比变化"""
    
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data = None
        self.intraday_data = None
        
    def fetch_daily_data(self, start_date: str = "2023-08-01", end_date: str = "2025-09-30"):
        """获取日线数据"""
        bs.login()
        
        rs = bs.query_history_k_data_plus(
            self.stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3"
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        bs.logout()
        
        if not data_list:
            print(f"未获取到 {self.stock_code} 的数据")
            return None
        
        self.data = pd.DataFrame(data_list, columns=rs.fields)
        
        # 数据类型转换
        numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.dropna()
        
        print(f"成功获取 {len(self.data)} 条日线数据")
        return self.data
    
    def calculate_opening_volume_ratio(self, period: int = 5):
        """计算开盘量比（基于前N日平均成交量）"""
        if self.data is None:
            return None
        
        volumes = self.data['volume'].values
        opening_volume_ratios = []
        
        for i in range(len(volumes)):
            if i < period:
                # 前几天数据不足，使用可用数据计算
                avg_volume = np.mean(volumes[:i+1]) if i > 0 else volumes[0]
            else:
                # 使用前N日平均成交量
                avg_volume = np.mean(volumes[i-period:i])
            
            # 开盘量比 = 当日成交量 / 前N日平均成交量
            volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
            opening_volume_ratios.append(volume_ratio)
        
        return np.array(opening_volume_ratios)
    
    def identify_significant_volume_days(self, volume_ratios: np.array, threshold: float = 2.0):
        """识别显著放量日（量比超过阈值）"""
        significant_days = []
        
        for i, ratio in enumerate(volume_ratios):
            if ratio >= threshold:
                date = self.data.iloc[i]['date']
                price = self.data.iloc[i]['close']
                pct_change = self.data.iloc[i]['pctChg']
                
                significant_days.append({
                    'date': date,
                    'volume_ratio': ratio,
                    'price': price,
                    'pct_change': pct_change,
                    'index': i
                })
        
        return significant_days
    
    def analyze_volume_price_relationship(self, volume_ratios: np.array):
        """分析量价关系"""
        if self.data is None:
            return None
        
        price_changes = self.data['pctChg'].values
        analysis_results = []
        
        # 分类分析
        for i in range(len(volume_ratios)):
            ratio = volume_ratios[i]
            pct_change = price_changes[i]
            
            # 量价关系分类
            if ratio >= 2.0:  # 显著放量
                if pct_change > 2.0:
                    relationship = "放量上涨"
                    signal_strength = "强"
                elif pct_change < -2.0:
                    relationship = "放量下跌"
                    signal_strength = "强"
                else:
                    relationship = "放量震荡"
                    signal_strength = "中"
            elif ratio >= 1.5:  # 温和放量
                if pct_change > 1.0:
                    relationship = "温和放量上涨"
                    signal_strength = "中"
                elif pct_change < -1.0:
                    relationship = "温和放量下跌"
                    signal_strength = "中"
                else:
                    relationship = "温和放量震荡"
                    signal_strength = "弱"
            else:  # 缩量
                if abs(pct_change) > 3.0:
                    relationship = "缩量异动"
                    signal_strength = "中"
                else:
                    relationship = "缩量整理"
                    signal_strength = "弱"
            
            analysis_results.append({
                'date': self.data.iloc[i]['date'],
                'volume_ratio': ratio,
                'pct_change': pct_change,
                'relationship': relationship,
                'signal_strength': signal_strength
            })
        
        return analysis_results
    
    def detect_breakout_with_volume(self, boxes: List[Dict], volume_ratios: np.array, 
                                   min_volume_ratio: float = 2.0):
        """结合箱体突破检测量比确认信号"""
        if self.data is None:
            return []
        
        prices = self.data['close'].values
        breakout_signals = []
        
        for box in boxes:
            start_idx = box['start_idx']
            end_idx = box['end_idx']
            
            # 检查箱体后的突破
            for i in range(end_idx + 1, min(len(prices), end_idx + 20)):  # 检查箱体后20天
                current_price = prices[i]
                current_volume_ratio = volume_ratios[i]
                
                # 向上突破
                if current_price > box['resistance'] and current_volume_ratio >= min_volume_ratio:
                    breakout_pct = (current_price - box['resistance']) / box['resistance'] * 100
                    
                    signal = {
                        'type': 'upward_breakout_with_volume',
                        'date': self.data.iloc[i]['date'],
                        'box_id': boxes.index(box),
                        'breakout_price': current_price,
                        'breakout_pct': breakout_pct,
                        'volume_ratio': current_volume_ratio,
                        'resistance_level': box['resistance'],
                        'support_level': box['support'],
                        'signal_strength': self._calculate_volume_signal_strength(
                            breakout_pct, current_volume_ratio)
                    }
                    breakout_signals.append(signal)
                    break
                
                # 向下突破
                elif current_price < box['support'] and current_volume_ratio >= min_volume_ratio:
                    breakout_pct = (box['support'] - current_price) / box['support'] * 100
                    
                    signal = {
                        'type': 'downward_breakout_with_volume',
                        'date': self.data.iloc[i]['date'],
                        'box_id': boxes.index(box),
                        'breakout_price': current_price,
                        'breakout_pct': breakout_pct,
                        'volume_ratio': current_volume_ratio,
                        'resistance_level': box['resistance'],
                        'support_level': box['support'],
                        'signal_strength': self._calculate_volume_signal_strength(
                            breakout_pct, current_volume_ratio)
                    }
                    breakout_signals.append(signal)
                    break
        
        return breakout_signals
    
    def _calculate_volume_signal_strength(self, breakout_pct: float, volume_ratio: float):
        """计算基于量比的信号强度"""
        base_strength = min(breakout_pct * 0.5, 5.0)  # 突破幅度贡献
        volume_strength = min((volume_ratio - 1) * 2, 10.0)  # 量比贡献
        
        total_strength = base_strength + volume_strength
        
        if total_strength >= 8:
            return "很强"
        elif total_strength >= 5:
            return "强"
        elif total_strength >= 3:
            return "中等"
        else:
            return "弱"
    
    def plot_volume_analysis(self, volume_ratios: np.array, significant_days: List[Dict],
                           figsize=(16, 12)):
        """绘制量比分析图表"""
        if self.data is None:
            return
        
        dates = pd.to_datetime(self.data['date']).dt.to_pydatetime()
        prices = self.data['close'].values
        volumes = self.data['volume'].values
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 价格走势图 + 显著放量日标记
        ax1.plot(dates, prices, 'b-', linewidth=1, label='收盘价')
        
        for day in significant_days:
            idx = day['index']
            ax1.scatter(dates[idx], prices[idx], 
                       color='red' if day['pct_change'] > 0 else 'green',
                       s=100, alpha=0.8, zorder=5)
            ax1.annotate(f"量比:{day['volume_ratio']:.1f}", 
                        (dates[idx], prices[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, ha='left')
        
        ax1.set_title(f'{self.stock_code} 价格走势与显著放量日', fontsize=12)
        ax1.set_ylabel('价格', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 量比走势图
        ax2.plot(dates, volume_ratios, 'orange', linewidth=1.5, label='量比')
        ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='量比1.5')
        ax2.axhline(y=2.0, color='red', linestyle='-', alpha=0.7, label='量比2.0')
        ax2.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
        
        # 标记显著放量日
        for day in significant_days:
            idx = day['index']
            ax2.scatter(dates[idx], volume_ratios[idx], 
                       color='red', s=80, alpha=0.8, zorder=5)
        
        ax2.set_title('量比走势', fontsize=12)
        ax2.set_ylabel('量比', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 成交量柱状图
        ax3.bar(dates, volumes, color='gray', alpha=0.6, width=0.8, label='成交量')
        
        # 标记显著放量日
        for day in significant_days:
            idx = day['index']
            ax3.bar(dates[idx], volumes[idx], 
                   color='red' if day['pct_change'] > 0 else 'green',
                   alpha=0.8, width=0.8)
        
        ax3.set_title('成交量分布', fontsize=12)
        ax3.set_ylabel('成交量', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 量价关系散点图
        pct_changes = self.data['pctChg'].values
        colors = ['red' if r >= 2.0 else 'orange' if r >= 1.5 else 'gray' 
                 for r in volume_ratios]
        
        scatter = ax4.scatter(volume_ratios, pct_changes, c=colors, alpha=0.6, s=30)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axvline(x=1.5, color='red', linestyle='--', alpha=0.7)
        ax4.axvline(x=2.0, color='red', linestyle='-', alpha=0.7)
        
        ax4.set_title('量价关系分析', fontsize=12)
        ax4.set_xlabel('量比', fontsize=10)
        ax4.set_ylabel('涨跌幅(%)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"volume_ratio_analysis_{self.stock_code}_{timestamp}"
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{filename}.jpg', dpi=300, bbox_inches='tight')
        print(f"量比分析图表已保存: {filename}.png/.jpg")
        
        plt.show()
    
    def print_volume_analysis_report(self, volume_ratios: np.array, 
                                   significant_days: List[Dict],
                                   analysis_results: List[Dict]):
        """打印量比分析报告"""
        print("\n" + "="*60)
        print(f"           {self.stock_code} 量比特征分析报告")
        print("="*60)
        
        # 基础统计
        print(f"\n📊 基础统计:")
        print(f"   分析周期: {len(self.data)} 个交易日")
        print(f"   平均量比: {np.mean(volume_ratios):.2f}")
        print(f"   最大量比: {np.max(volume_ratios):.2f}")
        print(f"   量比标准差: {np.std(volume_ratios):.2f}")
        
        # 显著放量日统计
        print(f"\n🔥 显著放量日统计 (量比≥2.0):")
        print(f"   显著放量日数: {len(significant_days)} 天")
        if significant_days:
            avg_ratio = np.mean([day['volume_ratio'] for day in significant_days])
            print(f"   平均量比: {avg_ratio:.2f}")
            
            up_days = [day for day in significant_days if day['pct_change'] > 0]
            down_days = [day for day in significant_days if day['pct_change'] < 0]
            
            print(f"   放量上涨: {len(up_days)} 天")
            print(f"   放量下跌: {len(down_days)} 天")
        
        # 量价关系统计
        print(f"\n📈 量价关系统计:")
        relationship_counts = {}
        for result in analysis_results:
            rel = result['relationship']
            relationship_counts[rel] = relationship_counts.get(rel, 0) + 1
        
        for relationship, count in sorted(relationship_counts.items(), 
                                        key=lambda x: x[1], reverse=True):
            percentage = count / len(analysis_results) * 100
            print(f"   {relationship}: {count} 天 ({percentage:.1f}%)")
        
        # 近期显著放量日详情
        print(f"\n📅 近期显著放量日详情:")
        recent_days = sorted(significant_days, key=lambda x: x['date'], reverse=True)[:10]
        
        for day in recent_days:
            date_str = day['date'].strftime('%Y-%m-%d')
            print(f"   {date_str}: 量比{day['volume_ratio']:.1f}, "
                  f"涨跌{day['pct_change']:+.2f}%, 价格{day['price']:.2f}")

def main():
    """主函数 - 演示量比分析功能"""
    # 分析000063
    analyzer = VolumeRatioAnalyzer("sz.000063")
    
    # 获取数据
    data = analyzer.fetch_daily_data()
    if data is None:
        return
    
    # 计算量比
    volume_ratios = analyzer.calculate_opening_volume_ratio()
    
    # 识别显著放量日
    significant_days = analyzer.identify_significant_volume_days(volume_ratios, threshold=2.0)
    
    # 分析量价关系
    analysis_results = analyzer.analyze_volume_price_relationship(volume_ratios)
    
    # 绘制分析图表
    analyzer.plot_volume_analysis(volume_ratios, significant_days)
    
    # 打印分析报告
    analyzer.print_volume_analysis_report(volume_ratios, significant_days, analysis_results)

if __name__ == "__main__":
    main()