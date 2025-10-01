"""
实战化箱体策略 - 基于市场结构和交易行为的箱体识别
作者: ZigZag策略团队
版本: 1.0
日期: 2025-01-27

核心理念：
1. 箱体必须基于真实的支撑阻力位
2. 考虑成交量分布和时间因素
3. 验证多次测试的有效性
4. 关注突破的确认信号
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PracticalBoxStrategy:
    """实战化箱体策略"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化策略
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        self.data = data.copy()
        self.data.reset_index(drop=True, inplace=True)
        
        # 计算技术指标
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        """计算必要的技术指标"""
        # 成交量移动平均
        self.data['volume_ma20'] = self.data['volume'].rolling(20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma20']
        
        # 价格波动率
        self.data['price_volatility'] = self.data['close'].rolling(20).std() / self.data['close'].rolling(20).mean()
        
        # ATR (平均真实波幅)
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.data['atr'] = true_range.rolling(14).mean()
        
    def identify_key_levels(self, swing_points: List[Dict]) -> List[Dict]:
        """
        识别关键价格位 - 基于实际交易行为
        
        Args:
            swing_points: ZigZag摆动点
            
        Returns:
            关键价格位列表
        """
        if len(swing_points) < 10:
            return []
            
        key_levels = []
        prices = [p['price'] for p in swing_points]
        
        # 1. 识别高频测试的价格位
        price_clusters = self._find_price_clusters(swing_points)
        
        for cluster in price_clusters:
            if len(cluster['points']) >= 3:  # 至少3次测试
                # 计算价格位的有效性
                effectiveness = self._calculate_level_effectiveness(cluster)
                
                if effectiveness['score'] > 0.6:  # 有效性阈值
                    key_levels.append({
                        'price': cluster['center_price'],
                        'type': 'key_level',
                        'strength': len(cluster['points']),
                        'effectiveness': effectiveness,
                        'points': cluster['points'],
                        'role': self._determine_level_role(cluster)
                    })
        
        return sorted(key_levels, key=lambda x: x['effectiveness']['score'], reverse=True)
    
    def _find_price_clusters(self, swing_points: List[Dict], tolerance: float = 0.015) -> List[Dict]:
        """
        寻找价格聚集区域
        
        Args:
            swing_points: 摆动点
            tolerance: 价格容忍度 (1.5%)
            
        Returns:
            价格聚集区域列表
        """
        clusters = []
        processed_points = set()
        
        for i, point in enumerate(swing_points):
            if i in processed_points:
                continue
                
            cluster_points = [point]
            cluster_indices = {i}
            base_price = point['price']
            
            # 寻找相近的价格点
            for j, other_point in enumerate(swing_points):
                if j in processed_points or j == i:
                    continue
                    
                price_diff = abs(other_point['price'] - base_price) / base_price
                if price_diff <= tolerance:
                    cluster_points.append(other_point)
                    cluster_indices.add(j)
            
            if len(cluster_points) >= 2:  # 至少2个点形成聚集
                processed_points.update(cluster_indices)
                
                prices = [p['price'] for p in cluster_points]
                clusters.append({
                    'center_price': np.mean(prices),
                    'price_range': max(prices) - min(prices),
                    'points': cluster_points,
                    'indices': cluster_indices
                })
        
        return clusters
    
    def _calculate_level_effectiveness(self, cluster: Dict) -> Dict:
        """
        计算价格位的有效性
        
        Args:
            cluster: 价格聚集区域
            
        Returns:
            有效性评估结果
        """
        points = cluster['points']
        center_price = cluster['center_price']
        
        # 1. 测试次数评分 (30%)
        test_count_score = min(1.0, len(points) / 5.0)
        
        # 2. 反弹/回调强度评分 (40%)
        reaction_strengths = []
        for point in points:
            # 计算从该点的反弹/回调幅度
            point_index = point['index']
            if point_index < len(self.data) - 5:
                if point['type'] == 'low':
                    # 计算反弹幅度
                    future_high = self.data['high'][point_index:point_index+10].max()
                    reaction_strength = (future_high - point['price']) / point['price']
                else:
                    # 计算回调幅度
                    future_low = self.data['low'][point_index:point_index+10].min()
                    reaction_strength = (point['price'] - future_low) / point['price']
                
                reaction_strengths.append(reaction_strength)
        
        avg_reaction = np.mean(reaction_strengths) if reaction_strengths else 0
        reaction_score = min(1.0, avg_reaction / 0.03)  # 3%反弹为满分
        
        # 3. 成交量确认评分 (20%)
        volume_confirmations = 0
        for point in points:
            point_index = point['index']
            if point_index < len(self.data):
                volume_ratio = self.data.loc[point_index, 'volume_ratio']
                if volume_ratio > 1.2:  # 成交量放大
                    volume_confirmations += 1
        
        volume_score = volume_confirmations / len(points)
        
        # 4. 时间分布评分 (10%)
        indices = [p['index'] for p in points]
        time_span = max(indices) - min(indices)
        time_score = min(1.0, time_span / 100)  # 跨越100个周期为满分
        
        # 综合评分
        total_score = (test_count_score * 0.3 + 
                      reaction_score * 0.4 + 
                      volume_score * 0.2 + 
                      time_score * 0.1)
        
        return {
            'score': total_score,
            'test_count': len(points),
            'avg_reaction': avg_reaction,
            'volume_confirmations': volume_confirmations,
            'time_span': time_span,
            'breakdown': {
                'test_count_score': test_count_score,
                'reaction_score': reaction_score,
                'volume_score': volume_score,
                'time_score': time_score
            }
        }
    
    def _determine_level_role(self, cluster: Dict) -> str:
        """
        确定价格位的角色 (支撑/阻力/转换)
        
        Args:
            cluster: 价格聚集区域
            
        Returns:
            价格位角色
        """
        points = cluster['points']
        high_count = sum(1 for p in points if p['type'] == 'high')
        low_count = sum(1 for p in points if p['type'] == 'low')
        
        if high_count > low_count * 1.5:
            return 'resistance'
        elif low_count > high_count * 1.5:
            return 'support'
        else:
            return 'support_resistance'  # 支撑阻力转换位
    
    def identify_trading_boxes(self, key_levels: List[Dict]) -> List[Dict]:
        """
        识别具有交易价值的箱体
        
        Args:
            key_levels: 关键价格位
            
        Returns:
            交易箱体列表
        """
        if len(key_levels) < 2:
            return []
            
        boxes = []
        
        # 寻找合适的支撑阻力位组合
        for i, resistance_level in enumerate(key_levels):
            for j, support_level in enumerate(key_levels):
                if i >= j:
                    continue
                    
                resistance_price = resistance_level['price']
                support_price = support_level['price']
                
                # 确保阻力位在支撑位之上
                if resistance_price <= support_price:
                    continue
                
                # 检查箱体的合理性
                box_height = resistance_price - support_price
                box_height_pct = box_height / support_price
                
                # 箱体高度应该在合理范围内 (2%-15%)
                if not (0.02 <= box_height_pct <= 0.15):
                    continue
                
                # 验证箱体的有效性
                box_validity = self._validate_box(resistance_level, support_level)
                
                if box_validity['is_valid']:
                    # 确定箱体的时间范围
                    time_range = self._determine_box_timeframe(resistance_level, support_level)
                    
                    boxes.append({
                        'type': 'trading_box',
                        'resistance_price': resistance_price,
                        'support_price': support_price,
                        'center_price': (resistance_price + support_price) / 2,
                        'box_height': box_height,
                        'box_height_pct': box_height_pct,
                        'resistance_level': resistance_level,
                        'support_level': support_level,
                        'validity': box_validity,
                        'time_range': time_range,
                        'trading_score': self._calculate_trading_score(resistance_level, support_level, box_validity),
                        'risk_reward_ratio': self._calculate_risk_reward(box_height_pct)
                    })
        
        # 按交易评分排序，返回最佳箱体
        return sorted(boxes, key=lambda x: x['trading_score'], reverse=True)[:3]
    
    def _validate_box(self, resistance_level: Dict, support_level: Dict) -> Dict:
        """
        验证箱体的有效性
        
        Args:
            resistance_level: 阻力位
            support_level: 支撑位
            
        Returns:
            验证结果
        """
        resistance_price = resistance_level['price']
        support_price = support_level['price']
        
        # 1. 检查价格位的强度
        min_strength = min(resistance_level['strength'], support_level['strength'])
        strength_valid = min_strength >= 2
        
        # 2. 检查有效性评分
        min_effectiveness = min(resistance_level['effectiveness']['score'], 
                               support_level['effectiveness']['score'])
        effectiveness_valid = min_effectiveness >= 0.5
        
        # 3. 检查角色匹配
        resistance_role_valid = resistance_level['role'] in ['resistance', 'support_resistance']
        support_role_valid = support_level['role'] in ['support', 'support_resistance']
        
        # 4. 检查时间重叠
        resistance_points = resistance_level['points']
        support_points = support_level['points']
        
        resistance_indices = set(p['index'] for p in resistance_points)
        support_indices = set(p['index'] for p in support_points)
        
        # 计算时间范围重叠度
        all_indices = resistance_indices.union(support_indices)
        min_index = min(all_indices)
        max_index = max(all_indices)
        overlap_range = max_index - min_index
        
        time_valid = overlap_range >= 20  # 至少跨越20个周期
        
        is_valid = (strength_valid and effectiveness_valid and 
                   resistance_role_valid and support_role_valid and time_valid)
        
        return {
            'is_valid': is_valid,
            'strength_valid': strength_valid,
            'effectiveness_valid': effectiveness_valid,
            'role_valid': resistance_role_valid and support_role_valid,
            'time_valid': time_valid,
            'min_strength': min_strength,
            'min_effectiveness': min_effectiveness,
            'overlap_range': overlap_range
        }
    
    def _determine_box_timeframe(self, resistance_level: Dict, support_level: Dict) -> Dict:
        """
        确定箱体的时间范围
        
        Args:
            resistance_level: 阻力位
            support_level: 支撑位
            
        Returns:
            时间范围信息
        """
        all_points = resistance_level['points'] + support_level['points']
        all_indices = [p['index'] for p in all_points]
        
        start_index = min(all_indices)
        end_index = max(all_indices)
        
        # 扩展箱体范围到最近的有效数据
        extended_end = min(len(self.data) - 1, end_index + 20)
        
        return {
            'start_index': start_index,
            'end_index': end_index,
            'extended_end': extended_end,
            'duration': end_index - start_index,
            'total_points': len(all_points)
        }
    
    def _calculate_trading_score(self, resistance_level: Dict, support_level: Dict, validity: Dict) -> float:
        """
        计算箱体的交易评分
        
        Args:
            resistance_level: 阻力位
            support_level: 支撑位
            validity: 有效性验证结果
            
        Returns:
            交易评分 (0-1)
        """
        # 基础评分
        base_score = 0.5
        
        # 强度加分
        avg_strength = (resistance_level['strength'] + support_level['strength']) / 2
        strength_bonus = min(0.2, avg_strength * 0.05)
        
        # 有效性加分
        avg_effectiveness = (resistance_level['effectiveness']['score'] + 
                           support_level['effectiveness']['score']) / 2
        effectiveness_bonus = avg_effectiveness * 0.2
        
        # 时间跨度加分
        time_bonus = min(0.1, validity['overlap_range'] / 1000)
        
        total_score = base_score + strength_bonus + effectiveness_bonus + time_bonus
        
        return min(1.0, total_score)
    
    def _calculate_risk_reward(self, box_height_pct: float) -> Dict:
        """
        计算风险收益比
        
        Args:
            box_height_pct: 箱体高度百分比
            
        Returns:
            风险收益信息
        """
        # 箱体内交易的风险收益比
        # 风险：箱体高度的一半
        # 收益：突破后的目标 (箱体高度)
        
        risk_pct = box_height_pct / 2
        reward_pct = box_height_pct
        
        risk_reward_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        return {
            'risk_pct': risk_pct,
            'reward_pct': reward_pct,
            'ratio': risk_reward_ratio,
            'quality': 'excellent' if risk_reward_ratio >= 2.5 else 
                      'good' if risk_reward_ratio >= 2.0 else 
                      'fair' if risk_reward_ratio >= 1.5 else 'poor'
        }
    
    def detect_breakout_signals(self, boxes: List[Dict]) -> List[Dict]:
        """
        检测箱体突破信号
        
        Args:
            boxes: 交易箱体列表
            
        Returns:
            突破信号列表
        """
        signals = []
        
        for box in boxes:
            resistance_price = box['resistance_price']
            support_price = box['support_price']
            time_range = box['time_range']
            
            # 在箱体时间范围内检测突破
            start_idx = max(0, time_range['start_index'])
            end_idx = min(len(self.data), time_range['extended_end'])
            
            for i in range(start_idx + 1, end_idx):
                current_price = self.data.loc[i, 'close']
                prev_price = self.data.loc[i-1, 'close']
                volume_ratio = self.data.loc[i, 'volume_ratio']
                
                # 向上突破检测
                if (current_price > resistance_price * 1.005 and  # 0.5%突破确认
                    prev_price <= resistance_price):
                    
                    # 突破确认条件
                    volume_confirm = volume_ratio > 1.3
                    price_confirm = current_price > resistance_price * 1.01  # 1%确认
                    
                    if volume_confirm and price_confirm:
                        signal_strength = self._calculate_breakout_strength(
                            box, 'upward', i, volume_ratio
                        )
                        
                        signals.append({
                            'index': i,
                            'type': 'buy',
                            'signal_type': 'box_breakout_up',
                            'price': current_price,
                            'box_info': box,
                            'strength': signal_strength,
                            'volume_confirm': volume_confirm,
                            'entry_price': current_price,
                            'stop_loss': support_price,
                            'take_profit': current_price + (resistance_price - support_price),
                            'risk_reward': box['risk_reward_ratio']
                        })
                
                # 向下突破检测
                elif (current_price < support_price * 0.995 and  # 0.5%突破确认
                      prev_price >= support_price):
                    
                    # 突破确认条件
                    volume_confirm = volume_ratio > 1.3
                    price_confirm = current_price < support_price * 0.99  # 1%确认
                    
                    if volume_confirm and price_confirm:
                        signal_strength = self._calculate_breakout_strength(
                            box, 'downward', i, volume_ratio
                        )
                        
                        signals.append({
                            'index': i,
                            'type': 'sell',
                            'signal_type': 'box_breakout_down',
                            'price': current_price,
                            'box_info': box,
                            'strength': signal_strength,
                            'volume_confirm': volume_confirm,
                            'entry_price': current_price,
                            'stop_loss': resistance_price,
                            'take_profit': current_price - (resistance_price - support_price),
                            'risk_reward': box['risk_reward_ratio']
                        })
        
        return signals
    
    def _calculate_breakout_strength(self, box: Dict, direction: str, index: int, volume_ratio: float) -> float:
        """
        计算突破信号强度
        
        Args:
            box: 箱体信息
            direction: 突破方向
            index: 突破位置
            volume_ratio: 成交量比率
            
        Returns:
            信号强度 (0-1)
        """
        base_strength = 0.6
        
        # 箱体质量加分
        trading_score_bonus = box['trading_score'] * 0.2
        
        # 成交量确认加分
        volume_bonus = min(0.2, (volume_ratio - 1.0) * 0.1)
        
        # 风险收益比加分
        rr_ratio = box['risk_reward_ratio']['ratio']
        rr_bonus = min(0.1, (rr_ratio - 1.5) * 0.05)
        
        total_strength = base_strength + trading_score_bonus + volume_bonus + rr_bonus
        
        return min(1.0, total_strength)