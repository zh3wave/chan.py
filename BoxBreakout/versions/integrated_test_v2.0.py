import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import baostock as bs
from scipy.signal import argrelextrema
from typing import List, Tuple, Dict, Optional
import warnings
import json
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_analyzer import EnhancedBoxBreakoutAnalyzer
from signal_validator import SignalValidator, BreakoutSignal, SignalType, ConfirmationLevel

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class IntegratedBoxBreakoutTest:
    """
    集成箱体突破测试系统
    结合原有测试代码的优点和新的验证机制
    """
    
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.analyzer = EnhancedBoxBreakoutAnalyzer(stock_code)
        self.validator = SignalValidator(
            macd_threshold=0.01,
            volume_ratio_threshold=1.5,
            breakout_threshold=0.5,
            confirmation_periods=3
        )
        self.test_results = {}
        
    def run_comprehensive_test(self, start_date: str = "2023-08-01", 
                             end_date: str = "2025-09-30"):
        """运行综合测试"""
        print(f"开始对 {self.stock_code} 进行综合箱体突破测试...")
        
        # 1. 获取数据
        print("1. 获取股票数据...")
        data = self.analyzer.fetch_data(start_date, end_date)
        if data is None:
            print("数据获取失败！")
            return None
            
        print(f"   成功获取 {len(data)} 条数据")
        
        # 2. 基础分析
        print("2. 执行基础技术分析...")
        prices = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values
        dates = pd.to_datetime(data['date'])
        
        # 计算技术指标
        macd_data = self.analyzer.calculate_macd(prices)
        volume_ratio = self.analyzer.calculate_volume_ratio(volumes)
        
        # ZigZag分析
        swing_points, zigzag_line = self.analyzer.zigzag_algorithm(prices)
        print(f"   识别摆动点: {len(swing_points)} 个")
        
        # 箱体识别
        boxes = self.analyzer.identify_boxes_with_indicators(
            prices, volumes, macd_data
        )
        print(f"   识别箱体: {len(boxes)} 个")
        
        # 3. 信号验证
        print("3. 执行突破信号验证...")
        validated_signals = self.validator.batch_validate_signals(
            boxes, prices, volumes, macd_data, dates.tolist()
        )
        print(f"   验证突破信号: {len(validated_signals)} 个")
        
        # 4. 生成验证报告
        print("4. 生成验证报告...")
        validation_report = self.validator.generate_signal_report(validated_signals)
        
        # 5. 保存测试结果
        self.test_results = {
            'stock_code': self.stock_code,
            'test_period': {'start': start_date, 'end': end_date},
            'data_summary': {
                'total_days': len(data),
                'price_range': {'min': float(prices.min()), 'max': float(prices.max())},
                'volume_range': {'min': float(volumes.min()), 'max': float(volumes.max())}
            },
            'technical_analysis': {
                'swing_points_count': len(swing_points),
                'boxes_count': len(boxes),
                'zigzag_points': len(zigzag_line)
            },
            'signal_validation': validation_report,
            'boxes': self._serialize_boxes(boxes),
            'signals': self._serialize_signals(validated_signals)
        }
        
        # 6. 绘制综合分析图
        print("5. 绘制综合分析图...")
        self._plot_comprehensive_analysis(
            data, prices, volumes, macd_data, volume_ratio,
            boxes, validated_signals, swing_points, zigzag_line
        )
        
        # 7. 打印详细结果
        self._print_comprehensive_results()
        
        return self.test_results
    
    def _serialize_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """序列化箱体数据"""
        serialized = []
        for box in boxes:
            serialized.append({
                'start_idx': int(box['start_idx']),
                'end_idx': int(box['end_idx']),
                'resistance': float(box['resistance']),
                'support': float(box['support']),
                'duration': int(box['duration']),
                'volatility': float(box['volatility']),
                'strength': float(box['strength']),
                'macd_score': float(box.get('macd_score', 0)),
                'volume_score': float(box.get('volume_score', 0))
            })
        return serialized
    
    def _serialize_signals(self, signals: List[BreakoutSignal]) -> List[Dict]:
        """序列化信号数据"""
        serialized = []
        for signal in signals:
            serialized.append({
                'signal_id': signal.signal_id,
                'signal_type': signal.signal_type.value,
                'timestamp': signal.timestamp.isoformat(),
                'price': float(signal.price),
                'breakout_percentage': float(signal.breakout_percentage),
                'confirmation_level': signal.confirmation_level.value,
                'signal_strength': float(signal.signal_strength),
                'macd_confirmed': signal.macd_confirmed,
                'volume_confirmed': signal.volume_confirmed,
                'volume_ratio': float(signal.volume_ratio),
                'risk_reward_ratio': float(signal.risk_reward_ratio) if signal.risk_reward_ratio != float('inf') else None,
                'follow_up_confirmed': signal.follow_up_confirmed,
                'max_favorable_move': float(signal.max_favorable_move) if signal.max_favorable_move else None,
                'max_adverse_move': float(signal.max_adverse_move) if signal.max_adverse_move else None,
                'fibonacci_targets': {k: float(v) for k, v in signal.fibonacci_targets.items()}
            })
        return serialized
    
    def _plot_comprehensive_analysis(self, data, prices, volumes, macd_data, 
                                   volume_ratio, boxes, signals, swing_points, zigzag_line):
        """绘制综合分析图"""
        dates = pd.to_datetime(data['date']).dt.to_pydatetime()
        
        # 创建子图 - 简化为2个分图
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        # 主图：K线 + 箱体 + 信号
        ax_main = fig.add_subplot(gs[0])
        self._plot_main_chart(ax_main, data, dates, prices, boxes, signals, zigzag_line)
        
        # 成交量图
        ax_volume = fig.add_subplot(gs[1])
        self._plot_volume_chart(ax_volume, dates, volumes, volume_ratio)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f'integrated_test_{self.stock_code}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # 确保charts目录存在
        charts_dir = 'charts'
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        # 保存到charts目录
        import os
        png_path = os.path.join(charts_dir, f'{filename}.png')
        jpg_path = os.path.join(charts_dir, f'{filename}.jpg')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(jpg_path, dpi=300, bbox_inches='tight')
        print(f"   综合分析图已保存到charts目录: {filename}.png/.jpg")
        
        plt.show()
    
    def _plot_main_chart(self, ax, data, dates, prices, boxes, signals, zigzag_line):
        """绘制主图表"""
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
            
            # 绘制箱体矩形
            rect = plt.Rectangle((start_date, box['support']), 
                               width_timedelta, height,
                               linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.2)
            ax.add_patch(rect)
            
            # 绘制关键区域的支撑阻力线（延伸到箱体后一段时间）
            extension_days = min(30, len(dates) - box['end_idx'] - 1)  # 最多延伸30天
            if extension_days > 0:
                extended_end_idx = min(box['end_idx'] + extension_days, len(dates) - 1)
                extended_end_date = dates[extended_end_idx]
                
                # 支撑线
                ax.plot([start_date, extended_end_date], 
                       [box['support'], box['support']], 
                       color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='支撑线' if i == 0 else "")
                
                # 阻力线
                ax.plot([start_date, extended_end_date], 
                       [box['resistance'], box['resistance']], 
                       color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='阻力线' if i == 0 else "")
            
            # 箱体标注
            mid_price = (box['resistance'] + box['support']) / 2
            mid_time = start_date + width_timedelta / 2
            ax.text(mid_time, mid_price, 
                    f'Box{i+1}\n强度:{box["strength"]:.1f}', 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 突破信号
        for signal in signals:
            if hasattr(signal, 'timestamp') and hasattr(signal, 'price'):
                # 找到对应的索引
                signal_date = pd.to_datetime(signal.timestamp)
                try:
                    idx = next(i for i, d in enumerate(dates) if pd.to_datetime(d).date() == signal_date.date())
                except StopIteration:
                    continue
                    
                if signal.signal_type == SignalType.UPWARD_BREAKOUT:
                    marker = '^'
                    color = 'red'
                else:
                    marker = 'v'
                    color = 'green'
                
                ax.scatter(dates[idx], signal.price, color=color, s=150, marker=marker, zorder=5)
                
                # 信号标注
                label = f"{signal.confirmation_level.value}\n强度:{signal.signal_strength:.2f}"
                ax.annotate(label, (dates[idx], signal.price), xytext=(10, 10), 
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax.set_title(f'{self.stock_code} 综合箱体突破分析', fontsize=16, fontweight='bold')
        ax.set_ylabel('价格', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_volume_chart(self, ax, dates, volumes, volume_ratio):
        """绘制成交量图"""
        ax.bar(dates, volumes, color='gray', alpha=0.6, width=0.8, label='成交量')
        
        ax2 = ax.twinx()
        ax2.plot(dates, volume_ratio, color='orange', linewidth=2, label='量比')
        ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='量比1.5')
        
        ax.set_title('成交量与量比', fontsize=12)
        ax.set_ylabel('成交量', fontsize=10)
        ax2.set_ylabel('量比', fontsize=10)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _print_comprehensive_results(self):
        """打印综合测试结果"""
        print(f"\n{'='*60}")
        print(f"  {self.stock_code} 综合箱体突破测试结果")
        print(f"{'='*60}")
        
        # 基础数据统计
        data_summary = self.test_results['data_summary']
        print(f"\n📊 数据概览:")
        print(f"   测试周期: {self.test_results['test_period']['start']} ~ {self.test_results['test_period']['end']}")
        print(f"   数据天数: {data_summary['total_days']} 天")
        print(f"   价格区间: {data_summary['price_range']['min']:.2f} ~ {data_summary['price_range']['max']:.2f}")
        
        # 技术分析结果
        tech_analysis = self.test_results['technical_analysis']
        print(f"\n🔍 技术分析:")
        print(f"   摆动点数量: {tech_analysis['swing_points_count']} 个")
        print(f"   识别箱体: {tech_analysis['boxes_count']} 个")
        print(f"   ZigZag节点: {tech_analysis['zigzag_points']} 个")
        
        # 信号验证结果
        validation = self.test_results['signal_validation']
        print(f"\n✅ 信号验证:")
        print(f"   总信号数: {validation['total_signals']} 个")
        if validation['total_signals'] > 0:
            print(f"   向上突破: {validation['signal_distribution']['upward_breakouts']} 个")
            print(f"   向下突破: {validation['signal_distribution']['downward_breakouts']} 个")
            print(f"   MACD确认率: {validation['validation_statistics']['macd_confirmation_rate']:.1f}%")
            print(f"   成交量确认率: {validation['validation_statistics']['volume_confirmation_rate']:.1f}%")
            print(f"   双重确认率: {validation['validation_statistics']['double_confirmation_rate']:.1f}%")
            
            if validation['follow_up_validation']['total_follow_up'] > 0:
                print(f"   后续验证成功率: {validation['follow_up_validation']['follow_up_success_rate']:.1f}%")
            
            # 平均指标
            avg_metrics = validation['average_metrics']
            print(f"\n📈 平均指标:")
            print(f"   平均信号强度: {avg_metrics['signal_strength']:.3f}")
            print(f"   平均突破幅度: {avg_metrics['breakout_percentage']:.2f}%")
            print(f"   平均量比: {avg_metrics['volume_ratio']:.2f}")
            print(f"   平均风险收益比: {avg_metrics['risk_reward_ratio']:.2f}")
            
            # 性能亮点
            performance = validation['performance_highlights']
            if performance['best_favorable_move']:
                print(f"\n🎯 性能亮点:")
                print(f"   最佳有利移动: {performance['best_favorable_move']:.2f}%")
                print(f"   最大不利移动: {performance['worst_adverse_move']:.2f}%")
        
        # 确认级别分布
        if 'confirmation_levels' in validation:
            print(f"\n🏆 确认级别分布:")
            for level, count in validation['confirmation_levels'].items():
                if count > 0:
                    print(f"   {level}: {count} 个")
        
        print(f"\n{'='*60}")
        
        # 保存结果到文件
        result_filename = f'test_results_{self.stock_code}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"📁 测试结果已保存到: {result_filename}")


def main():
    """主函数"""
    print("开始集成箱体突破测试...")
    
    # 创建测试实例
    tester = IntegratedBoxBreakoutTest("sz.000063")
    
    # 运行综合测试
    results = tester.run_comprehensive_test("2023-08-01", "2025-09-30")
    
    if results:
        print("\n✅ 集成测试完成！")
    else:
        print("\n❌ 集成测试失败！")

if __name__ == "__main__":
    main()