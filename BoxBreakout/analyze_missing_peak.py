"""
分析为什么修复后的ZigZag算法仍然遗漏真正的最高点
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 添加路径
sys.path.append('..')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_missing_peak():
    """分析遗漏最高点的原因"""
    
    # 加载数据
    try:
        data = pd.read_csv('../ETH_USDT_5m.csv')
        data['date'] = pd.to_datetime(data['date'])
        print(f"✅ 成功加载数据: {len(data)} 条记录")
    except:
        print("❌ 无法加载数据文件")
        return
    
    # 取最近1000条数据进行分析
    test_data = data.tail(1000).copy()
    test_data.reset_index(drop=True, inplace=True)
    
    highs = test_data['high'].values
    lows = test_data['low'].values
    
    # 找到真正的最高点
    actual_highest_idx = np.argmax(highs)
    actual_highest_price = highs[actual_highest_idx]
    
    print(f"\n🎯 真正的最高点分析:")
    print(f"   位置: K线#{actual_highest_idx}")
    print(f"   价格: ${actual_highest_price:.2f}")
    
    # 分析周围的价格
    start_idx = max(0, actual_highest_idx - 10)
    end_idx = min(len(highs), actual_highest_idx + 11)
    
    print(f"\n📊 最高点周围价格分析 (K线#{start_idx}-{end_idx-1}):")
    for i in range(start_idx, end_idx):
        marker = " 👑 " if i == actual_highest_idx else "    "
        print(f"   K线#{i:3d}: ${highs[i]:7.2f}{marker}")
    
    # 测试不同order参数的识别结果
    print(f"\n🔍 不同order参数的识别结果:")
    for order in [1, 2, 3, 4, 5]:
        high_indices = argrelextrema(highs, np.greater, order=order)[0]
        
        # 检查是否包含真正的最高点
        contains_peak = actual_highest_idx in high_indices
        status = "✅" if contains_peak else "❌"
        
        print(f"   order={order}: 识别{len(high_indices):3d}个高点 {status}")
        
        if not contains_peak and len(high_indices) > 0:
            # 找到最接近的高点
            closest_idx = min(high_indices, key=lambda x: abs(x - actual_highest_idx))
            closest_price = highs[closest_idx]
            print(f"            最接近的高点: K线#{closest_idx}, ${closest_price:.2f}")
    
    # 分析为什么order=2仍然遗漏
    print(f"\n🔬 详细分析order=2为什么遗漏最高点:")
    order = 2
    
    # 检查最高点前后各2个点的价格
    if actual_highest_idx >= order and actual_highest_idx < len(highs) - order:
        print(f"   检查条件: highs[{actual_highest_idx}] > 前后各{order}个点")
        
        all_greater = True
        for offset in range(-order, order + 1):
            if offset == 0:
                continue
            idx = actual_highest_idx + offset
            is_greater = highs[actual_highest_idx] > highs[idx]
            status = "✅" if is_greater else "❌"
            print(f"   ${highs[actual_highest_idx]:.2f} > ${highs[idx]:.2f} (K线#{idx}) {status}")
            if not is_greater:
                all_greater = False
        
        if all_greater:
            print(f"   🤔 所有条件都满足，但仍未被识别，可能是边界问题")
        else:
            print(f"   💡 不满足argrelextrema的条件，需要调整算法")
    
    # 绘制详细分析图
    plt.figure(figsize=(15, 10))
    
    # 子图1: 整体价格走势
    plt.subplot(2, 1, 1)
    plt.plot(range(len(highs)), highs, 'g-', linewidth=1, label='最高价')
    plt.scatter(actual_highest_idx, actual_highest_price, 
               color='red', marker='*', s=200, zorder=10, 
               label=f'真正最高点 ${actual_highest_price:.2f}')
    
    # 标记不同order参数识别的高点
    colors = ['blue', 'orange', 'purple', 'brown', 'pink']
    for i, order in enumerate([1, 2, 3, 4, 5]):
        high_indices = argrelextrema(highs, np.greater, order=order)[0]
        if len(high_indices) > 0:
            plt.scatter(high_indices, highs[high_indices], 
                       color=colors[i], marker='o', s=30, alpha=0.7,
                       label=f'order={order} ({len(high_indices)}个点)')
    
    plt.title('不同order参数的高点识别对比')
    plt.xlabel('K线索引')
    plt.ylabel('价格 ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 最高点周围的详细视图
    plt.subplot(2, 1, 2)
    zoom_start = max(0, actual_highest_idx - 20)
    zoom_end = min(len(highs), actual_highest_idx + 21)
    zoom_range = range(zoom_start, zoom_end)
    
    plt.plot(zoom_range, highs[zoom_start:zoom_end], 'g-', linewidth=2, marker='o')
    plt.scatter(actual_highest_idx, actual_highest_price, 
               color='red', marker='*', s=300, zorder=10)
    
    # 标记order=2识别的点
    high_indices_2 = argrelextrema(highs, np.greater, order=2)[0]
    zoom_high_indices = [idx for idx in high_indices_2 if zoom_start <= idx < zoom_end]
    if zoom_high_indices:
        plt.scatter(zoom_high_indices, highs[zoom_high_indices], 
                   color='blue', marker='^', s=100, alpha=0.8, label='order=2识别的高点')
    
    plt.title(f'最高点周围详细视图 (K线#{zoom_start}-{zoom_end-1})')
    plt.xlabel('K线索引')
    plt.ylabel('价格 ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加价格标注
    for i in zoom_range:
        if i % 5 == 0 or i == actual_highest_idx:  # 每5个点标注一次，加上最高点
            plt.annotate(f'${highs[i]:.1f}', 
                        (i, highs[i]), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = 'missing_peak_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 分析图表已保存: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    analyze_missing_peak()