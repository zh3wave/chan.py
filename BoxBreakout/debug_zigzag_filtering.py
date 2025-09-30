"""
调试ZigZag算法的过滤逻辑
找出为什么argrelextrema能识别到最高点，但ZigZag算法却遗漏了
"""

import pandas as pd
import numpy as np
import sys
from scipy.signal import argrelextrema

# 添加路径
sys.path.append('..')

def debug_zigzag_filtering():
    """调试ZigZag算法的过滤逻辑"""
    
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
    
    print(f"\n🎯 真正的最高点: K线#{actual_highest_idx}, 价格${actual_highest_price:.2f}")
    
    # 模拟ZigZag算法的完整流程
    deviation = 1.0
    order = 2
    
    print(f"\n🔍 ZigZag算法调试 (deviation={deviation}%, order={order}):")
    print("=" * 60)
    
    # 步骤1: 使用argrelextrema识别极值点
    high_indices = argrelextrema(highs, np.greater, order=order)[0]
    low_indices = argrelextrema(lows, np.less, order=order)[0]
    
    print(f"步骤1 - argrelextrema识别:")
    print(f"   高点数量: {len(high_indices)}")
    print(f"   低点数量: {len(low_indices)}")
    print(f"   包含真正最高点: {'✅' if actual_highest_idx in high_indices else '❌'}")
    
    if actual_highest_idx in high_indices:
        high_idx_position = np.where(high_indices == actual_highest_idx)[0][0]
        print(f"   真正最高点在高点列表中的位置: #{high_idx_position}")
    
    # 步骤2: 合并并排序极值点
    all_extrema = []
    for idx in high_indices:
        all_extrema.append((idx, highs[idx], 'high'))
    for idx in low_indices:
        all_extrema.append((idx, lows[idx], 'low'))
    
    all_extrema.sort(key=lambda x: x[0])
    
    print(f"\n步骤2 - 合并排序:")
    print(f"   合并后极值点总数: {len(all_extrema)}")
    
    # 找到真正最高点在合并列表中的位置
    highest_in_merged = None
    for i, point in enumerate(all_extrema):
        if point[0] == actual_highest_idx:
            highest_in_merged = i
            break
    
    if highest_in_merged is not None:
        print(f"   真正最高点在合并列表中的位置: #{highest_in_merged}")
        print(f"   前一个点: {all_extrema[highest_in_merged-1] if highest_in_merged > 0 else 'None'}")
        print(f"   当前点: {all_extrema[highest_in_merged]}")
        print(f"   后一个点: {all_extrema[highest_in_merged+1] if highest_in_merged < len(all_extrema)-1 else 'None'}")
    
    # 步骤3: 过滤逻辑详细分析
    print(f"\n步骤3 - 过滤逻辑详细分析:")
    print("=" * 40)
    
    filtered_points = [all_extrema[0]] if all_extrema else []
    removed_points = []
    
    for i, current in enumerate(all_extrema[1:], 1):
        last = filtered_points[-1] if filtered_points else None
        
        if last is None:
            continue
            
        # 计算价格变化幅度
        price_change_pct = abs(current[1] - last[1]) / last[1] * 100
        
        print(f"\n处理点#{i}: K线#{current[0]}, {current[2]}, ${current[1]:.2f}")
        print(f"   上一个保留点: K线#{last[0]}, {last[2]}, ${last[1]:.2f}")
        print(f"   价格变化: {price_change_pct:.2f}% (阈值: {deviation}%)")
        
        if price_change_pct >= deviation:
            print(f"   ✅ 价格变化足够大")
            
            # 如果类型相同，保留更极端的点
            if current[2] == last[2]:
                print(f"   🔄 类型相同({current[2]})，比较极值:")
                if (current[2] == 'high' and current[1] > last[1]) or \
                   (current[2] == 'low' and current[1] < last[1]):
                    print(f"   🔄 替换: ${last[1]:.2f} -> ${current[1]:.2f}")
                    removed_points.append(filtered_points[-1])
                    filtered_points[-1] = current
                else:
                    print(f"   ❌ 保留原点: ${last[1]:.2f}")
                    removed_points.append(current)
            else:
                print(f"   ✅ 类型不同，添加新点")
                filtered_points.append(current)
        else:
            print(f"   ❌ 价格变化不够大，跳过")
            removed_points.append(current)
        
        # 特别关注真正的最高点
        if current[0] == actual_highest_idx:
            print(f"   🎯 这是真正的最高点！")
            if current not in filtered_points:
                print(f"   ⚠️  真正的最高点被过滤掉了！")
    
    print(f"\n📊 过滤结果:")
    print(f"   保留的摆动点: {len(filtered_points)}")
    print(f"   被移除的点: {len(removed_points)}")
    
    # 检查真正的最高点是否被保留
    highest_preserved = any(point[0] == actual_highest_idx for point in filtered_points)
    print(f"   真正最高点是否保留: {'✅' if highest_preserved else '❌'}")
    
    if not highest_preserved:
        print(f"\n🔍 分析真正最高点被移除的原因:")
        for point in removed_points:
            if point[0] == actual_highest_idx:
                print(f"   真正最高点被移除: {point}")
                break
    
    # 显示最终的摆动点
    print(f"\n📍 最终保留的摆动点:")
    for i, point in enumerate(filtered_points[:10]):  # 只显示前10个
        marker = " 🎯" if point[0] == actual_highest_idx else ""
        date_str = test_data['date'].iloc[point[0]].strftime('%m-%d %H:%M')
        print(f"   #{i+1:2d}: {point[2]:4s} | K线#{point[0]:3d} | ${point[1]:7.2f} | {date_str}{marker}")

if __name__ == "__main__":
    debug_zigzag_filtering()