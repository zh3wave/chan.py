#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试量比优先逻辑的交易信号识别
验证17日、5日、27日等关键交易时机
"""

import pandas as pd
import numpy as np
from support_resistance_analyzer import SupportResistanceAnalyzer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import baostock as bs

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fetch_stock_data(stock_code: str, start_date: str = "2023-08-01", end_date: str = "2025-01-31"):
    """获取股票数据"""
    # 登录baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f'登录失败: {lg.error_msg}')
        return None
    
    # 获取日K线数据
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3"
    )
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    # 登出baostock
    bs.logout()
    
    if not data_list:
        print("未获取到数据")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 数据类型转换
    numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['date'] = pd.to_datetime(df['date'])
    
    # 过滤掉停牌日期
    df = df[df['tradestatus'] == '1'].copy()
    
    return df

def test_structure_first_logic():
    """测试先结构后量比的交易逻辑"""
    
    # 获取数据
    print("获取股票数据...")
    data = fetch_stock_data("sz.000001", "2023-01-01", "2025-01-31")
    if data is None:
        print("数据获取失败")
        return None
    
    # 创建分析器
    analyzer = SupportResistanceAnalyzer("sz.000001")
    analyzer.data = data
    
    # 第一步：识别分水线（支撑阻力线）
    highs = data['high'].values
    lows = data['low'].values
    swing_points = analyzer.zigzag_algorithm(highs, lows)
    key_levels = analyzer.identify_key_levels(swing_points)
    
    print("=== 分水线识别结果 ===")
    print(f"识别到 {len(swing_points)} 个转折点")
    print(f"识别到 {len(key_levels)} 条分水线")
    
    for i, level in enumerate(key_levels[:10]):  # 显示前10条
        print(f"分水线{i+1}: 价格 {level['price']:.2f}, 触及{level['touches']}次, 强度{level['strength']:.1f}")
    
    # 第二步：识别结构突破
    structure_breakouts = []
    prices = data['close'].values
    dates = data['date'].values
    
    for i in range(len(prices)):
        current_price = prices[i]
        current_date = dates[i]
        
        # 检查是否突破任何分水线
        for level in key_levels:
            level_price = level['price']
            
            # 向上突破检查（突破阻力）
            if current_price > level_price * 1.01:  # 1%突破阈值
                # 确认突破：检查前一天是否在分水线下方
                if i > 0 and prices[i-1] <= level_price:
                    structure_breakouts.append({
                        'date': current_date,
                        'price': current_price,
                        'level_price': level_price,
                        'breakout_type': 'upward',
                        'strength': (current_price - level_price) / level_price,
                        'level_touches': level['touches']
                    })
            
            # 向下突破检查（跌破支撑）
            elif current_price < level_price * 0.99:  # 1%跌破阈值
                # 确认突破：检查前一天是否在分水线上方
                if i > 0 and prices[i-1] >= level_price:
                    structure_breakouts.append({
                        'date': current_date,
                        'price': current_price,
                        'level_price': level_price,
                        'breakout_type': 'downward',
                        'strength': (level_price - current_price) / level_price,
                        'level_touches': level['touches']
                    })
    
    print(f"\n=== 结构突破识别结果 ===")
    print(f"识别到 {len(structure_breakouts)} 个结构突破")
    
    # 第三步：对结构突破进行分时量比确认
    # 注意：这里暂时用日线量比模拟分时量比，实际应该获取分时数据
    confirmed_signals = []
    
    for breakout in structure_breakouts:
        breakout_date = pd.to_datetime(breakout['date'])
        # 找到对应的数据索引
        date_mask = data['date'] == breakout_date
        if date_mask.any():
            idx = data[date_mask].index[0]
            # 计算当日量比（这里用日线量比代替分时量比）
            volume_ratio = analyzer._calculate_volume_ratio(data['volume'].values, idx)
            
            # 分时量比确认（阈值可以调整）
            if volume_ratio > 1.2:  # 分时量比阈值
                confirmed_signals.append({
                    **breakout,
                    'volume_ratio': volume_ratio,
                    'confirmed': True
                })
                
                direction = "向上突破" if breakout['breakout_type'] == 'upward' else "向下突破"
                date_str = breakout_date.strftime('%Y-%m-%d')
                print(f"✓ {date_str}: {direction} 分水线{breakout['level_price']:.2f}, "
                      f"突破强度{breakout['strength']:.1%}, 量比{volume_ratio:.2f}")
    
    print(f"\n=== 最终确认信号 ===")
    print(f"结构突破后量比确认的信号: {len(confirmed_signals)} 个")
    
    # 重点关注的日期
    key_dates = ['2024-12-17', '2024-11-05', '2024-09-25', '2024-09-27']
    
    print("\n=== 重点关注日期分析 ===")
    for key_date in key_dates:
        key_datetime = pd.to_datetime(key_date)
        found = False
        
        for signal in confirmed_signals:
            if pd.to_datetime(signal['date']).date() == key_datetime.date():
                direction = "向上突破" if signal['breakout_type'] == 'upward' else "向下突破"
                print(f"✓ {key_date}: {direction}, 量比 {signal['volume_ratio']:.2f}")
                found = True
                break
        
        if not found:
            # 查看该日期是否有结构突破但量比未达标
            date_index = data[data['date'].dt.date == key_datetime.date()].index
            if len(date_index) > 0:
                idx = date_index[0]
                current_price = data['close'].iloc[idx]
                
                # 检查是否有结构突破
                has_structure_breakout = False
                for breakout in structure_breakouts:
                    if pd.to_datetime(breakout['date']).date() == key_datetime.date():
                        has_structure_breakout = True
                        volume_ratio = analyzer._calculate_volume_ratio(data['volume'].values, idx)
                        direction = "向上突破" if breakout['breakout_type'] == 'upward' else "向下突破"
                        print(f"△ {key_date}: {direction} 但量比{volume_ratio:.2f}未达标")
                        break
                
                if not has_structure_breakout:
                    print(f"✗ {key_date}: 无结构突破")
            else:
                print(f"✗ {key_date}: 数据中未找到该日期")
    
    # 绘制分析图表
    plot_structure_analysis(data, key_levels, structure_breakouts, confirmed_signals, key_dates)
    
    return confirmed_signals

def plot_structure_analysis(data, key_levels, structure_breakouts, confirmed_signals, key_dates):
    """绘制先结构后量比的分析图表"""
    
    # 按照附件布局：行情图占50%，成交量和量比各占25%
    fig = plt.figure(figsize=(15, 12))
    
    # 创建子图：行情图占50%高度，成交量和量比各占25%
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)  # 行情图占2行（50%）
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1)  # 成交量占1行（25%）
    ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)  # 量比占1行（25%）
    
    # 价格走势图 + 分水线
    ax1.plot(data['date'], data['close'], 'b-', linewidth=1, alpha=0.7, label='收盘价')
    
    # 绘制分水线（这是关键！）
    for i, level in enumerate(key_levels):
        # 根据分水线对应的zigzag点范围来绘制，而不是延伸到右侧
        start_idx = level.get('start_idx', 0)
        end_idx = level.get('end_idx', start_idx + 30)  # 使用分水线实际的结束位置
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(data) - 1, end_idx)
        
        start_date = data['date'].iloc[start_idx]
        end_date = data['date'].iloc[end_idx]
        
        # 根据强度设置线条样式
        if level['strength'] > 3:
            linestyle = '-'
            alpha = 0.8
            linewidth = 2
        else:
            linestyle = '--'
            alpha = 0.6
            linewidth = 1.5
        
        # 绘制分水线
        ax1.hlines(level['price'], start_date, end_date, 
                  colors='purple', linestyles=linestyle, alpha=alpha, linewidth=linewidth)
        
        # 标注分水线信息
        ax1.text(end_date, level['price'], 
                f"分水线{level['touches']}次", 
                fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='purple', alpha=0.3))
    
    # 标记结构突破点
    for breakout in structure_breakouts:
        color = 'orange' if breakout['breakout_type'] == 'upward' else 'red'
        marker = '^' if breakout['breakout_type'] == 'upward' else 'v'
        ax1.scatter(breakout['date'], breakout['price'], color=color, s=60, marker=marker, alpha=0.7, zorder=4)
    
    # 标记最终确认信号
    for signal in confirmed_signals:
        color = 'lime' if signal['breakout_type'] == 'upward' else 'magenta'
        marker = '↑' if signal['breakout_type'] == 'upward' else '↓'
        ax1.scatter(signal['date'], signal['price'], color=color, s=100, marker='o', zorder=5)
        ax1.annotate(f"{marker}{signal['volume_ratio']:.1f}", 
                    (signal['date'], signal['price']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=color, weight='bold')
    
    # 标记重点关注日期
    for key_date in key_dates:
        key_datetime = pd.to_datetime(key_date)
        date_data = data[data['date'].dt.date == key_datetime.date()]
        if len(date_data) > 0:
            ax1.axvline(x=key_datetime, color='green', linestyle='--', alpha=0.7)
            ax1.text(key_datetime, date_data['close'].iloc[0], key_date, 
                    rotation=90, verticalalignment='bottom', fontsize=8, color='green')
    
    ax1.set_title('价格走势、分水线与突破信号')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 成交量图 - 调整显示比例和样式
    # 将成交量转换为万手单位以便显示
    volume_in_wan = data['volume'] / 10000
    ax2.bar(data['date'], volume_in_wan, alpha=0.6, color='gray', width=1)
    
    # 标记确认信号的成交量
    for signal in confirmed_signals:
        color = 'lime' if signal['breakout_type'] == 'upward' else 'magenta'
        volume_data = data[data['date'] == signal['date']]['volume']
        if len(volume_data) > 0:
            ax2.bar(signal['date'], volume_data.iloc[0] / 10000, color=color, alpha=0.8, width=1)
    
    ax2.set_title('成交量与量比确认')
    ax2.set_ylabel('成交量(万手)')
    ax2.grid(True, alpha=0.3)
    
    # 量比图 - 优化显示范围和样式
    volume_ratios = []
    dates = []
    
    for i in range(5, len(data)):
        analyzer = SupportResistanceAnalyzer("sz.000001")
        volume_ratio = analyzer._calculate_volume_ratio(data['volume'].values, i)
        volume_ratios.append(volume_ratio)
        dates.append(data['date'].iloc[i])
    
    ax3.plot(dates, volume_ratios, 'b-', linewidth=1.5, alpha=0.8, label='开盘量比')
    ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='量比确认线 1.2')
    ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='强势量比线 1.5')
    ax3.axhline(y=2.0, color='purple', linestyle='--', alpha=0.8, linewidth=2, label='超强量比线 2.0')
    
    # 标记确认信号的量比
    for signal in confirmed_signals:
        color = 'lime' if signal['breakout_type'] == 'upward' else 'magenta'
        ax3.scatter(signal['date'], signal['volume_ratio'], color=color, s=80, alpha=0.9, zorder=5, 
                   edgecolors='black', linewidth=1)
        # 添加量比数值标注
        ax3.annotate(f'{signal["volume_ratio"]:.1f}', 
                    (signal['date'], signal['volume_ratio']), 
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=8, color=color, weight='bold', ha='center')
    
    # 设置量比图的y轴范围，突出重要区域
    ax3.set_ylim(0.5, max(3.0, max(volume_ratios) * 1.1))
    ax3.set_title('开盘量比对比')
    ax3.set_ylabel('量比')
    ax3.set_xlabel('日期')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('structure_first_logic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    confirmed_signals = test_structure_first_logic()