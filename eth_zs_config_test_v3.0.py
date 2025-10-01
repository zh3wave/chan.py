"""
ETH缠论中枢配置测试脚本 V3.0
功能：测试三种不同中枢配置参数对ETH数据分析结果的影响
版本：V3.0 - 支持三种配置类型，K线数量增加到800根
创建时间：2025年
"""

from Chan import CChan
from ChanConfig import CChanConfig
from ZS.ZSConfig import CZSConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.PlotDriver import CPlotDriver
import pandas as pd
import os
from datetime import datetime

def load_eth_data_segment(csv_file, start_index=200000, segment_length=800):
    """
    从ETH CSV文件中加载指定段的数据
    
    Args:
        csv_file: CSV文件路径
        start_index: 起始索引（固定200000）
        segment_length: 数据段长度（增加到800根）
    
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

def get_zs_config_params(config_type="strict_theory"):
    """
    获取不同类型的中枢配置参数
    
    Args:
        config_type: 配置类型 "strict_theory"/"fine_analysis"/"cross_segment"
    
    Returns:
        dict: 中枢配置参数字典
    """
    if config_type == "strict_theory":
        # 严格缠论配置（传统理论）
        return {
            "zs_combine": False,      # 不合并中枢
            "zs_combine_mode": "zs",  # 使用zs模式
            "one_bi_zs": False,       # 不允许单笔中枢
            "zs_algo": "normal"       # 段内中枢算法
        }
    elif config_type == "fine_analysis":
        # 精细分析配置（趋势分析）
        return {
            "zs_combine": True,       # 合并中枢
            "zs_combine_mode": "zs",  # 使用zs模式合并
            "one_bi_zs": True,        # 允许单笔中枢
            "zs_algo": "normal"       # 段内中枢（必须）
        }
    elif config_type == "cross_segment":
        # 跨段分析配置（不支持单笔中枢）
        return {
            "zs_combine": True,       # 合并中枢
            "zs_combine_mode": "peak", # 使用peak模式合并
            "one_bi_zs": False,       # 必须为False
            "zs_algo": "over_seg"     # 跨段中枢算法
        }
    else:
        raise ValueError(f"未知的配置类型: {config_type}")

def get_config_description(config_type):
    """获取配置类型的中文描述"""
    descriptions = {
        "strict_theory": "严格缠论配置（传统理论）",
        "fine_analysis": "精细分析配置（趋势分析）", 
        "cross_segment": "跨段分析配置"
    }
    return descriptions.get(config_type, config_type)

def analyze_with_config(config_type="strict_theory"):
    """
    使用指定配置分析ETH数据
    
    Args:
        config_type: 配置类型
    """
    print(f"\n{'='*60}")
    print(f"🔧 开始 {get_config_description(config_type)} 分析")
    print(f"{'='*60}")
    
    # ETH数据文件路径
    eth_file = "ETH_USDT_5m.csv"
    
    # 检查文件是否存在
    if not os.path.exists(eth_file):
        print(f"❌ ETH数据文件不存在: {eth_file}")
        return False
    
    # 加载ETH数据段（增加到800根K线）
    eth_data = load_eth_data_segment(eth_file, start_index=200000, segment_length=800)
    
    if eth_data is None:
        print("❌ 无法加载ETH数据")
        return False
    
    # 获取中枢配置参数
    zs_params = get_zs_config_params(config_type)
    print(f"📋 中枢配置:")
    print(f"   - zs_combine: {zs_params['zs_combine']}")
    print(f"   - zs_combine_mode: {zs_params['zs_combine_mode']}")
    print(f"   - one_bi_zs: {zs_params['one_bi_zs']}")
    print(f"   - zs_algo: {zs_params['zs_algo']}")
    
    # 配置CChan参数，将中枢参数直接加入配置字典
    config_dict = {
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
    }
    
    # 添加中枢配置参数
    config_dict.update(zs_params)
    
    config = CChanConfig(config_dict)

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
            "x_range": 800,  # 显示全部800根K线
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
        
        # 统计笔信息
        bi_list = chan.kl_datas[KL_TYPE.K_5M].bi_list
        print(f"✏️ 笔数量: {len(bi_list)}")
        
        # 统计线段信息
        seg_list = chan.kl_datas[KL_TYPE.K_5M].seg_list
        print(f"📏 线段数量: {len(seg_list)}")
        
        # 统计中枢信息
        zs_list = chan.kl_datas[KL_TYPE.K_5M].zs_list
        print(f"🎯 中枢数量: {len(zs_list)}")
        if len(zs_list) > 0:
            print(f"📊 中枢详情:")
            for i, zs in enumerate(zs_list):
                zs_type = "单笔中枢" if zs.is_one_bi_zs() else "多笔中枢"
                print(f"   中枢{i+1}: {zs.begin_bi.idx}->{zs.end_bi.idx}, 范围: {zs.low:.2f}-{zs.high:.2f} ({zs_type})")
        
        # 统计买卖点信息
        bsp_list = chan.kl_datas[KL_TYPE.K_5M].bs_point_lst
        print(f"📍 买卖点数量: {len(bsp_list)}")
        
        # 创建绘图驱动器
        print("\n🎨 开始绘制图表...")
        plot_driver = CPlotDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
        
        # 显示图表
        plot_driver.figure.show()
        
        # 确保eth_charts目录存在
        os.makedirs("eth_charts", exist_ok=True)
        
        # 保存图表到eth_charts目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"eth_charts/eth_zs_{config_type}_config_800k_{timestamp}.png"
        plot_driver.save2img(output_file)
        print(f"💾 图表已保存: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试三种配置
    configs = [
        ("strict_theory", "严格缠论配置（传统理论）"),
        ("fine_analysis", "精细分析配置（趋势分析）"),
        ("cross_segment", "跨段分析配置")
    ]
    
    print("🚀 开始ETH缠论中枢配置对比测试")
    print(f"📊 K线数量: 800根（比之前增加300根）")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success_count = 0
    
    for config_type, description in configs:
        print(f"\n🔄 正在测试: {description}")
        success = analyze_with_config(config_type)
        if not success:
            print(f"❌ {description} 分析失败")
        else:
            print(f"✅ {description} 分析完成")
            success_count += 1
        print("\n" + "="*60)
    
    print(f"\n🎉 测试完成！成功生成 {success_count}/{len(configs)} 个配置的图表")
    print("📁 所有图表已保存到 eth_charts/ 目录")