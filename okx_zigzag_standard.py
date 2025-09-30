"""
OKX标准ZigZag指标实现
基于用户提供的OKX参数：deviation=5, depth=10
用户测试发现deviation=1效果很好
"""

import numpy as np
from scipy.signal import argrelextrema


class OKXZigZag:
    """OKX标准ZigZag指标实现"""
    
    def __init__(self, deviation: float = 5.0, depth: int = 10):
        """
        初始化OKX ZigZag指标
        
        参数：
        - deviation: 最小价格变动百分比，默认5.0%（OKX标准）
        - depth: 寻找局部极值的深度，默认10（OKX标准）
        
        注：用户测试发现deviation=1效果很好
        """
        self.deviation = deviation
        self.depth = depth
    
    def calculate(self, highs: np.array, lows: np.array):
        """
        计算ZigZag指标
        
        参数：
        - highs: 最高价数组
        - lows: 最低价数组
        
        返回：
        - swing_points: 摆动点列表
        - zigzag_line: ZigZag线数据
        """
        swing_points = []
        zigzag_line = []
        
        if len(highs) < self.depth or len(lows) < self.depth:
            return swing_points, zigzag_line
        
        # 使用更小的order参数来识别更多的局部极值点
        # 原来的order=self.depth//2可能过于严格，改为固定值2
        order = 2  # 更敏感的极值点识别
        high_indices = argrelextrema(highs, np.greater, order=order)[0]
        low_indices = argrelextrema(lows, np.less, order=order)[0]
        
        # 合并并排序极值点
        all_extrema = []
        for idx in high_indices:
            all_extrema.append((idx, highs[idx], 'high'))
        for idx in low_indices:
            all_extrema.append((idx, lows[idx], 'low'))
        
        all_extrema.sort(key=lambda x: x[0])
        
        if not all_extrema:
            return swing_points, zigzag_line
        
        # 过滤掉幅度不够的摆动，但要特别处理相同类型的连续极值点
        filtered_points = [all_extrema[0]] if all_extrema else []
        
        for current in all_extrema[1:]:
            if not filtered_points:
                filtered_points.append(current)
                continue
                
            last = filtered_points[-1]
            
            # 计算价格变化幅度
            price_change_pct = abs(current[1] - last[1]) / last[1] * 100
            
            if price_change_pct >= self.deviation:
                # 如果类型相同，保留更极端的点
                if current[2] == last[2]:
                    if (current[2] == 'high' and current[1] > last[1]) or \
                       (current[2] == 'low' and current[1] < last[1]):
                        filtered_points[-1] = current
                else:
                    filtered_points.append(current)
            else:
                # 即使价格变化不够大，如果是相同类型的更极端点，也要替换
                if current[2] == last[2]:
                    if (current[2] == 'high' and current[1] > last[1]) or \
                       (current[2] == 'low' and current[1] < last[1]):
                        filtered_points[-1] = current
        
        # 转换为摆动点格式
        for point in filtered_points:
            swing_points.append({
                'index': point[0],
                'price': point[1],
                'type': point[2]
            })
            zigzag_line.append((point[0], point[1]))
        
        return swing_points, zigzag_line
    
    def get_parameters(self):
        """获取当前参数设置"""
        return {
            'deviation': self.deviation,
            'depth': self.depth
        }
    
    def set_parameters(self, deviation: float = None, depth: int = None):
        """设置参数"""
        if deviation is not None:
            self.deviation = deviation
        if depth is not None:
            self.depth = depth


# 使用示例
if __name__ == "__main__":
    # 创建OKX标准ZigZag指标
    zigzag = OKXZigZag(deviation=5.0, depth=10)
    
    # 用户推荐的高敏感度设置
    zigzag_sensitive = OKXZigZag(deviation=1.0, depth=10)
    
    print("OKX标准ZigZag参数：", zigzag.get_parameters())
    print("用户推荐高敏感度参数：", zigzag_sensitive.get_parameters())