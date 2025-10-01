# E:\ChanClaude\baostock_data_fetch.py
import baostock as bs
import pandas as pd
from datetime import datetime
import os

def fetch_stock_data(code="sz.002050", start_date="2024-04-01", end_date="2024-04-30"):
    """
    获取股票历史数据
    
    Args:
        code: 股票代码，如 "sz.002050"
        start_date: 开始日期 "YYYY-MM-DD"
        end_date: 结束日期 "YYYY-MM-DD"
    
    Returns:
        DataFrame: 股票数据
    """
    # 登录 Baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败，错误码: {lg.error_code}, 错误信息: {lg.error_msg}")
        return None
    
    print(f"开始查询股票 {code} 从 {start_date} 到 {end_date} 的数据...")
    
    # 查询历史K线数据
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus",
        start_date,
        end_date,
        frequency="d",  # 日K线
        adjustflag="3"  # 前复权
    )
    
    # 检查查询结果
    if rs.error_code != '0':
        print(f"查询失败，错误码: {rs.error_code}, 错误信息: {rs.error_msg}")
        bs.logout()
        return None
    
    # 获取数据
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    
    # 检查是否有数据
    if not data_list:
        print("未查询到数据！")
        bs.logout()
        return None
    
    # 创建 DataFrame
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 数据类型转换（添加错误处理）
    try:
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 转换换手率
        if 'turn' in df.columns:
            df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
            
    except Exception as e:
        print(f"数据类型转换出错: {e}")
    
    # 计算技术指标
    if len(df) >= 5:
        # 5日均量
        df['volume_5d_mean'] = df['volume'].rolling(window=5).mean().shift(1)
        # 量比
        df['volume_ratio'] = (df['volume'] / df['volume_5d_mean']).fillna(0).round(2)
    
    # 计算涨跌幅
    df['prev_close'] = df['close'].shift(1)
    df['change_percent'] = ((df['close'] - df['prev_close']) / df['prev_close'] * 100).round(2)
    df['change_amount'] = (df['close'] - df['prev_close']).round(2)
    
    # 保存到 CSV
    output_dir = "E:\\ChanClaude\\data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成文件名
    stock_code_clean = code.replace(".", "_")
    date_suffix = start_date.replace("-", "") + "_" + end_date.replace("-", "")
    output_file = os.path.join(output_dir, f"{stock_code_clean}_{date_suffix}_data.csv")
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"数据已保存到: {output_file}")
    print(f"共获取 {len(df)} 条数据")
    
    # 登出
    bs.logout()
    return df

def analyze_stock_basic_info(df, code):
    """分析股票基本信息"""
    if df is None or len(df) == 0:
        return
    
    print(f"\n=== {code} 基本信息分析 ===")
    print(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"期间最高价: {df['high'].max():.2f}")
    print(f"期间最低价: {df['low'].min():.2f}")
    print(f"期间涨跌幅: {df['change_percent'].sum():.2f}%")
    print(f"平均成交量: {df['volume'].mean():.0f}")
    print(f"平均成交额: {df['amount'].mean():.0f}")
    
    # 找出涨跌幅最大的几天
    max_up = df.loc[df['change_percent'].idxmax()]
    max_down = df.loc[df['change_percent'].idxmin()]
    
    print(f"\n最大涨幅日: {max_up['date']}, 涨幅: {max_up['change_percent']:.2f}%")
    print(f"最大跌幅日: {max_down['date']}, 跌幅: {max_down['change_percent']:.2f}%")

if __name__ == "__main__":
    # 执行查询
    code = "sz.002050"  # 三花智控
    data = fetch_stock_data(code)
    
    if data is not None:
        print("\n查询结果预览：")
        print(data[['date', 'open', 'high', 'low', 'close', 'volume', 'change_percent']].head(10))
        
        # 分析基本信息
        analyze_stock_basic_info(data, code)
        
        print(f"\n数据列名: {list(data.columns)}")
        print(f"数据形状: {data.shape}")
    else:
        print("数据获取失败！")