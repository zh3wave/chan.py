from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.PlotDriver import CPlotDriver
import pandas as pd
import os

def load_eth_data_segment(csv_file, start_index=10000, segment_length=500):
    """
    从ETH CSV文件中加载指定段的数据
    
    Args:
        csv_file: CSV文件路径
        start_index: 起始索引
        segment_length: 数据段长度
    
    Returns:
        选定的数据段
    """
    try:
        # 读取完整数据
        data = pd.read_csv(csv_file)
        print(f"✅ 成功加载ETH数据，总共 {len(data)} 条记录")
        
        # 选择指定段的数据
        end_index = start_index + segment_length
        if end_index > len(data):
            end_index = len(data)
            start_index = end_index - segment_length
            
        selected_data = data.iloc[start_index:end_index].copy()
        
        print(f"📊 选择数据段: 第 {start_index} 到 {end_index-1} 条记录")
        print(f"📅 时间范围: {selected_data['date'].iloc[0]} 到 {selected_data['date'].iloc[-1]}")
        print(f"💰 价格范围: {selected_data['low'].min():.2f} - {selected_data['high'].max():.2f}")
        
        return selected_data
        
    except Exception as e:
        print(f"❌ 加载ETH数据失败: {e}")
        return None

if __name__ == "__main__":
    # ETH数据文件路径
    eth_file = "ETH_USDT_5m.csv"
    
    # 检查文件是否存在
    if not os.path.exists(eth_file):
        print(f"❌ ETH数据文件不存在: {eth_file}")
        exit(1)
    
    # 加载ETH数据段（选择第5段：索引200000-200499）
    eth_data = load_eth_data_segment(eth_file, start_index=200000, segment_length=500)
    
    if eth_data is None:
        print("❌ 无法加载ETH数据")
        exit(1)
    
    # 配置CChan参数
    config = CChanConfig({
        "bi_strict": True,
        "trigger_step": False,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    # 绘图配置
    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": False,
        "plot_zs": True,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": True,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    plot_para = {
        "seg": {
            # "plot_trendline": True,
        },
        "bi": {
            # "show_num": True,
            # "disp_end": True,
        },
        "figure": {
            "x_range": 500,  # 显示全部500根K线
        },
    }
    
    try:
        print("\n🚀 开始创建CChan实例...")
        
        # 创建CChan实例，使用CSV数据源
        chan = CChan(
            code="ETH_USDT_5m",  # 代码名称
            begin_time=eth_data['date'].iloc[0],     # 使用选定数据段的开始时间
            end_time=eth_data['date'].iloc[-1],      # 使用选定数据段的结束时间
            data_src=DATA_SRC.CSV,  # 使用CSV数据源
            lv_list=[KL_TYPE.K_5M], # 5分钟K线
            config=config,
            autype=AUTYPE.QFQ,
        )
        
        print("✅ CChan实例创建成功")
        print(f"📈 K线数量: {len(chan.kl_datas[KL_TYPE.K_5M])}")
        
        # 创建绘图驱动器
        print("\n🎨 开始绘制图表...")
        plot_driver = CPlotDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
        
        # 显示图表
        plot_driver.figure.show()
        
        # 保存图表到eth_charts目录
        output_file = "eth_charts/eth_chan_segment_5.png"
        plot_driver.save2img(output_file)
        print(f"💾 图表已保存: {output_file}")
        
        # 清理临时文件（如果存在）
        temp_csv = "temp_eth_500k.csv"
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            print(f"🗑️ 临时文件已删除: {temp_csv}")
            
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        # 清理临时文件（如果存在）
        temp_csv = "temp_eth_500k.csv"
        if os.path.exists(temp_csv):
            os.remove(temp_csv)