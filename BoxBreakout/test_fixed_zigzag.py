"""
测试修复后的ZigZag算法
验证是否能正确识别用户指出的最高点
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加路径
sys.path.append('..')

from okx_zigzag_standard import OKXZigZag
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def test_zigzag_fix():
    """测试修复后的ZigZag算法"""
    
    # 加载数据
    try:
        data = pd.read_csv('../ETH_USDT_5m.csv')
        data['date'] = pd.to_datetime(data['date'])
        print(f"✅ 成功加载数据: {len(data)} 条记录")
    except:
        print("❌ 无法加载数据文件")
        return
    
    # 取最近1000条数据进行测试
    test_data = data.tail(1000).copy()
    test_data.reset_index(drop=True, inplace=True)
    
    highs = test_data['high'].values
    lows = test_data['low'].values
    closes = test_data['close'].values
    
    # 使用修复后的ZigZag算法
    zigzag = OKXZigZag(deviation=1.0, depth=10)
    swing_points, zigzag_line = zigzag.calculate(highs, lows)
    
    print(f"\n🔍 修复后的ZigZag算法结果:")
    print(f"   识别到 {len(swing_points)} 个摆动点")
    
    # 找到最高点和最低点
    high_points = [p for p in swing_points if p['type'] == 'high']
    low_points = [p for p in swing_points if p['type'] == 'low']
    
    if high_points:
        highest_point = max(high_points, key=lambda x: x['price'])
        print(f"   最高摆动点: K线#{highest_point['index']}, 价格${highest_point['price']:.2f}")
        
        # 检查是否是真正的最高点
        actual_highest_idx = np.argmax(highs)
        actual_highest_price = highs[actual_highest_idx]
        print(f"   实际最高点: K线#{actual_highest_idx}, 价格${actual_highest_price:.2f}")
        
        if highest_point['index'] == actual_highest_idx:
            print("   ✅ 成功识别到真正的最高点!")
        else:
            print("   ❌ 未能识别到真正的最高点")
            print(f"   差异: {abs(highest_point['price'] - actual_highest_price):.2f}")
    
    # 绘制图表验证
    plt.figure(figsize=(15, 8))
    
    # 绘制K线图（简化版）
    plt.plot(range(len(closes)), closes, 'k-', linewidth=0.8, alpha=0.7, label='收盘价')
    plt.plot(range(len(highs)), highs, 'g-', linewidth=0.5, alpha=0.5, label='最高价')
    plt.plot(range(len(lows)), lows, 'r-', linewidth=0.5, alpha=0.5, label='最低价')
    
    # 绘制ZigZag线
    if zigzag_line:
        zz_x = [point[0] for point in zigzag_line]
        zz_y = [point[1] for point in zigzag_line]
        plt.plot(zz_x, zz_y, 'b-', linewidth=2, alpha=0.8, label='ZigZag线')
    
    # 标记摆动点
    for i, point in enumerate(swing_points):
        color = 'red' if point['type'] == 'high' else 'blue'
        marker = '^' if point['type'] == 'high' else 'v'
        plt.scatter(point['index'], point['price'], 
                   color=color, marker=marker, s=50, zorder=5)
        
        # 只标记前20个点，避免图表过于拥挤
        if i < 20:
            plt.annotate(f"{i+1}", 
                        (point['index'], point['price']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=color)
    
    # 标记真正的最高点
    actual_highest_idx = np.argmax(highs)
    actual_highest_price = highs[actual_highest_idx]
    plt.scatter(actual_highest_idx, actual_highest_price, 
               color='orange', marker='*', s=200, zorder=10, 
               label=f'真正最高点 ${actual_highest_price:.2f}')
    
    plt.title('修复后的ZigZag算法测试 - 最近1000条数据', fontsize=14)
    plt.xlabel('K线索引')
    plt.ylabel('价格 ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    output_file = 'fixed_zigzag_test.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 图表已保存: {output_file}")
    
    plt.show()
    
    return swing_points

if __name__ == "__main__":
    swing_points = test_zigzag_fix()