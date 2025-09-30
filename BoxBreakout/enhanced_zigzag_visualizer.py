#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版ZigZag摆动点可视化分析器
专门用于清晰显示所有ZigZag摆动点的位置和编号

作者: AI Assistant
版本: V1.0
日期: 2025年
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_zigzag_standard import OKXZigZag

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class EnhancedZigZagVisualizer:
    """增强版ZigZag摆动点可视化分析器"""
    
    def __init__(self):
        self.data = None
        self.swing_points = []
        
    def load_data_segment(self, file_path: str, start_index: int = 200000, length: int = 800) -> bool:
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
    
    def analyze_with_zigzag(self, deviation: float = 1.0, depth: int = 10):
        """使用ZigZag参数进行分析"""
        highs = self.data['high'].values
        lows = self.data['low'].values
        
        # 使用OKX ZigZag算法
        zigzag = OKXZigZag(deviation=deviation, depth=depth)
        self.swing_points, zigzag_line = zigzag.calculate(highs, lows)
        
        print(f"🔍 ZigZag分析 (deviation={deviation}%, depth={depth}):")
        print(f"   识别摆动点: {len(self.swing_points)} 个")
        
        # 打印所有摆动点详情
        print(f"\n📍 摆动点详细信息:")
        for i, point in enumerate(self.swing_points):
            date_str = self.data['date'].iloc[point['index']].strftime('%m-%d %H:%M')
            print(f"   #{i+1:2d}: {point['type']:4s} | K线#{point['index']:3d} | ${point['price']:7.2f} | {date_str}")
        
        return {
            'swing_points': len(self.swing_points),
            'swing_details': self.swing_points
        }
    
    def create_enhanced_swing_chart(self, title_suffix: str = "", save_path: str = None):
        """创建增强版摆动点显示图表"""
        fig, ax = plt.subplots(1, 1, figsize=(24, 12))
        
        # 准备数据
        dates = self.data['date']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        closes = self.data['close']
        
        # 绘制K线 - 使用更细的线条
        for i in range(len(self.data)):
            color = 'red' if closes.iloc[i] >= opens.iloc[i] else 'green'
            alpha = 0.6
            
            # 绘制影线
            ax.plot([i, i], [lows.iloc[i], highs.iloc[i]], color='gray', linewidth=0.8, alpha=alpha)
            
            # 绘制实体
            body_height = abs(closes.iloc[i] - opens.iloc[i])
            body_bottom = min(opens.iloc[i], closes.iloc[i])
            
            rect = Rectangle((i-0.4, body_bottom), 0.8, body_height, 
                           facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.3)
            ax.add_patch(rect)
        
        # 绘制ZigZag摆动点 - 超级增强显示
        if self.swing_points:
            # 先绘制连线
            swing_x = [p['index'] for p in self.swing_points]
            swing_y = [p['price'] for p in self.swing_points]
            ax.plot(swing_x, swing_y, 'purple', linewidth=4, alpha=0.8, label='ZigZag连线', zorder=3)
            
            # 再绘制摆动点
            for i, point in enumerate(self.swing_points):
                color = 'darkblue' if point['type'] == 'high' else 'darkorange'
                marker = '^' if point['type'] == 'high' else 'v'
                
                # 超大摆动点标记
                ax.scatter(point['index'], point['price'], color=color, s=400, 
                          marker=marker, zorder=6, edgecolors='black', linewidth=3)
                
                # 添加白色内核
                ax.scatter(point['index'], point['price'], color='white', s=150, 
                          marker=marker, zorder=7, edgecolors='black', linewidth=1)
                
                # 摆动点编号和详细信息
                point_type_cn = '高点' if point['type'] == 'high' else '低点'
                date_str = self.data['date'].iloc[point['index']].strftime('%m-%d %H:%M')
                point_text = f"摆动点 #{i+1}\n{point_type_cn}\n${point['price']:.2f}\n{date_str}\nK线#{point['index']}"
                
                # 动态调整标注位置
                offset_y = 60 if point['type'] == 'high' else -80
                
                ax.annotate(point_text, 
                           (point['index'], point['price']), 
                           xytext=(0, offset_y),
                           textcoords='offset points', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.6', facecolor=color, alpha=0.9, 
                                   edgecolor='black', linewidth=2),
                           ha='center', va='center', zorder=8,
                           arrowprops=dict(arrowstyle='->', color='black', lw=2))
                
                # 添加垂直辅助线
                ax.axvline(x=point['index'], color=color, linestyle='--', alpha=0.7, linewidth=2, zorder=2)
                
                # 在底部添加K线索引标记
                ax.text(point['index'], ax.get_ylim()[0], f"K#{point['index']}", 
                       rotation=90, ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        # 设置图表属性
        ax.set_title(f'ZigZag摆动点详细标注图 - 用户推荐参数 (deviation=1.0%, depth=10){title_suffix}', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('价格 ($)', fontsize=14, fontweight='bold')
        ax.set_xlabel('K线索引', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 设置x轴标签 - 显示更多时间点
        step = max(1, len(self.data) // 20)
        ax.set_xticks(range(0, len(self.data), step))
        ax.set_xticklabels([f"{i}\n{dates.iloc[i].strftime('%m-%d %H:%M')}" 
                           for i in range(0, len(self.data), step)], 
                          rotation=45, fontsize=9)
        
        # 添加统计信息文本框
        stats_text = f"""摆动点统计信息:
• 总摆动点数: {len(self.swing_points)} 个
• 高点数量: {len([p for p in self.swing_points if p['type'] == 'high'])} 个  
• 低点数量: {len([p for p in self.swing_points if p['type'] == 'low'])} 个
• ZigZag参数: deviation=1.0%, depth=10
• 数据范围: 800根K线 (5分钟级别)"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
               verticalalignment='top', fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 增强摆动点图表已保存: {save_path}")
        
        plt.show()

def main():
    """主函数"""
    print("🚀 增强版ZigZag摆动点可视化分析器启动")
    
    # 初始化分析器
    visualizer = EnhancedZigZagVisualizer()
    
    # 加载ETH数据
    eth_file = "../ETH_USDT_5m.csv"
    if not visualizer.load_data_segment(eth_file, start_index=200000, length=800):
        return
    
    print(f"\n{'='*80}")
    print(f"📊 分析配置: 用户推荐参数")
    print(f"   参数: deviation=1.0%, depth=10")
    print('='*80)
    
    # 执行分析
    result = visualizer.analyze_with_zigzag(deviation=1.0, depth=10)
    
    # 生成增强版摆动点图表
    save_path = "enhanced_zigzag_swing_points_详细标注.png"
    visualizer.create_enhanced_swing_chart("", save_path)
    
    print(f"\n✅ 增强版摆动点可视化完成！")
    print(f"📈 总共识别到 {result['swing_points']} 个摆动点")
    print(f"📊 详细标注图表已保存: {save_path}")
    print(f"💡 现在您可以清楚地看到每个摆动点的:")
    print(f"   - 精确位置和编号")
    print(f"   - 价格和时间信息") 
    print(f"   - K线索引")
    print(f"   - 高点/低点类型")

if __name__ == "__main__":
    main()