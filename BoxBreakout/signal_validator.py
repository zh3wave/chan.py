import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    """信号类型枚举"""
    UPWARD_BREAKOUT = "upward_breakout"
    DOWNWARD_BREAKOUT = "downward_breakout"

class ConfirmationLevel(Enum):
    """确认级别枚举"""
    WEAK = "weak"           # 仅价格突破
    MODERATE = "moderate"   # 价格 + 单一指标确认
    STRONG = "strong"       # 价格 + 双重指标确认
    VERY_STRONG = "very_strong"  # 价格 + 双重指标 + 额外条件

@dataclass
class BreakoutSignal:
    """突破信号数据类"""
    signal_id: str
    signal_type: SignalType
    timestamp: pd.Timestamp
    price: float
    box_resistance: float
    box_support: float
    breakout_percentage: float
    
    # MACD验证
    macd_dif: float
    macd_dea: float
    macd_histogram: float
    macd_confirmed: bool
    macd_strength: float
    
    # 成交量验证
    volume: float
    volume_ratio: float
    volume_ma: float
    volume_confirmed: bool
    volume_strength: float
    
    # 综合评分
    confirmation_level: ConfirmationLevel
    signal_strength: float
    risk_reward_ratio: float
    
    # 斐波那契目标位
    fibonacci_targets: Dict[str, float]
    
    # 后续验证
    follow_up_confirmed: Optional[bool] = None
    max_favorable_move: Optional[float] = None
    max_adverse_move: Optional[float] = None

class SignalValidator:
    """
    突破信号验证器
    实现MACD+成交量双重验证体系
    """
    
    def __init__(self, 
                 macd_threshold: float = 0.01,
                 volume_ratio_threshold: float = 1.5,
                 breakout_threshold: float = 0.5,
                 confirmation_periods: int = 3):
        """
        初始化验证器
        
        Args:
            macd_threshold: MACD确认阈值
            volume_ratio_threshold: 量比确认阈值
            breakout_threshold: 突破幅度阈值(%)
            confirmation_periods: 确认周期数
        """
        self.macd_threshold = macd_threshold
        self.volume_ratio_threshold = volume_ratio_threshold
        self.breakout_threshold = breakout_threshold
        self.confirmation_periods = confirmation_periods
        
    def validate_breakout_signal(self, 
                                box: Dict,
                                price_data: np.array,
                                volume_data: np.array,
                                macd_data: Dict,
                                breakout_idx: int,
                                dates: List[pd.Timestamp]) -> Optional[BreakoutSignal]:
        """
        验证单个突破信号
        
        Args:
            box: 箱体信息
            price_data: 价格数据
            volume_data: 成交量数据
            macd_data: MACD数据
            breakout_idx: 突破点索引
            dates: 日期数据
            
        Returns:
            验证后的突破信号或None
        """
        if breakout_idx >= len(price_data):
            return None
            
        current_price = price_data[breakout_idx]
        resistance = box['resistance']
        support = box['support']
        
        # 判断突破方向
        if current_price > resistance:
            signal_type = SignalType.UPWARD_BREAKOUT
            breakout_pct = (current_price - resistance) / resistance * 100
            reference_level = resistance
        elif current_price < support:
            signal_type = SignalType.DOWNWARD_BREAKOUT
            breakout_pct = (support - current_price) / support * 100
            reference_level = support
        else:
            return None  # 未突破
            
        # 检查突破幅度
        if breakout_pct < self.breakout_threshold:
            return None
            
        # MACD验证
        macd_result = self._validate_macd(macd_data, breakout_idx, signal_type)
        
        # 成交量验证
        volume_result = self._validate_volume(volume_data, breakout_idx)
        
        # 计算综合强度和确认级别
        confirmation_level, signal_strength = self._calculate_confirmation_level(
            breakout_pct, macd_result, volume_result
        )
        
        # 计算斐波那契目标位
        fibonacci_targets = self._calculate_fibonacci_targets(
            resistance, support, signal_type
        )
        
        # 计算风险收益比
        risk_reward_ratio = self._calculate_risk_reward_ratio(
            current_price, reference_level, fibonacci_targets, signal_type
        )
        
        # 创建信号对象
        signal = BreakoutSignal(
            signal_id=f"{signal_type.value}_{breakout_idx}_{int(dates[breakout_idx].timestamp())}",
            signal_type=signal_type,
            timestamp=dates[breakout_idx],
            price=current_price,
            box_resistance=resistance,
            box_support=support,
            breakout_percentage=breakout_pct,
            
            macd_dif=macd_data['dif'][breakout_idx],
            macd_dea=macd_data['dea'][breakout_idx],
            macd_histogram=macd_data['macd'][breakout_idx],
            macd_confirmed=macd_result['confirmed'],
            macd_strength=macd_result['strength'],
            
            volume=volume_data[breakout_idx],
            volume_ratio=volume_result['ratio'],
            volume_ma=volume_result['ma'],
            volume_confirmed=volume_result['confirmed'],
            volume_strength=volume_result['strength'],
            
            confirmation_level=confirmation_level,
            signal_strength=signal_strength,
            risk_reward_ratio=risk_reward_ratio,
            fibonacci_targets=fibonacci_targets
        )
        
        return signal
    
    def _validate_macd(self, macd_data: Dict, idx: int, signal_type: SignalType) -> Dict:
        """验证MACD指标"""
        if idx >= len(macd_data['dif']) or idx >= len(macd_data['dea']):
            return {'confirmed': False, 'strength': 0.0}
            
        dif = macd_data['dif'][idx]
        dea = macd_data['dea'][idx]
        macd_hist = macd_data['macd'][idx]
        
        confirmed = False
        strength = 0.0
        
        if signal_type == SignalType.UPWARD_BREAKOUT:
            # 向上突破：DIF > DEA 且 MACD > 0
            if dif > dea and macd_hist > self.macd_threshold:
                confirmed = True
                strength = min(abs(dif - dea) + abs(macd_hist), 1.0)
                
                # 额外加分：DIF刚刚上穿DEA
                if idx > 0 and macd_data['dif'][idx-1] <= macd_data['dea'][idx-1]:
                    strength += 0.3
                    
        else:  # DOWNWARD_BREAKOUT
            # 向下突破：DIF < DEA 且 MACD < 0
            if dif < dea and macd_hist < -self.macd_threshold:
                confirmed = True
                strength = min(abs(dif - dea) + abs(macd_hist), 1.0)
                
                # 额外加分：DIF刚刚下穿DEA
                if idx > 0 and macd_data['dif'][idx-1] >= macd_data['dea'][idx-1]:
                    strength += 0.3
        
        return {
            'confirmed': confirmed,
            'strength': min(strength, 1.0)
        }
    
    def _validate_volume(self, volume_data: np.array, idx: int, ma_period: int = 5) -> Dict:
        """验证成交量指标"""
        if idx < ma_period:
            return {'confirmed': False, 'strength': 0.0, 'ratio': 1.0, 'ma': volume_data[idx]}
            
        # 计算成交量均线
        volume_ma = np.mean(volume_data[idx-ma_period:idx])
        current_volume = volume_data[idx]
        
        # 计算量比
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
        
        # 验证成交量放大
        confirmed = volume_ratio >= self.volume_ratio_threshold
        
        # 计算强度（量比越大强度越高，但有上限）
        if confirmed:
            strength = min((volume_ratio - 1.0) / 2.0, 1.0)
        else:
            strength = max(0.0, (volume_ratio - 1.0) / 2.0)
            
        return {
            'confirmed': confirmed,
            'strength': strength,
            'ratio': volume_ratio,
            'ma': volume_ma
        }
    
    def _calculate_confirmation_level(self, breakout_pct: float, 
                                    macd_result: Dict, 
                                    volume_result: Dict) -> Tuple[ConfirmationLevel, float]:
        """计算确认级别和信号强度"""
        base_strength = min(breakout_pct / 5.0, 1.0)  # 突破幅度基础分
        macd_strength = macd_result['strength'] if macd_result['confirmed'] else 0.0
        volume_strength = volume_result['strength'] if volume_result['confirmed'] else 0.0
        
        # 综合强度计算
        signal_strength = (base_strength * 0.4 + 
                          macd_strength * 0.3 + 
                          volume_strength * 0.3)
        
        # 确认级别判断
        if macd_result['confirmed'] and volume_result['confirmed']:
            if signal_strength >= 0.8:
                confirmation_level = ConfirmationLevel.VERY_STRONG
            else:
                confirmation_level = ConfirmationLevel.STRONG
        elif macd_result['confirmed'] or volume_result['confirmed']:
            confirmation_level = ConfirmationLevel.MODERATE
        else:
            confirmation_level = ConfirmationLevel.WEAK
            
        return confirmation_level, signal_strength
    
    def _calculate_fibonacci_targets(self, resistance: float, support: float, 
                                   signal_type: SignalType) -> Dict[str, float]:
        """计算斐波那契目标位"""
        box_height = resistance - support
        
        if signal_type == SignalType.UPWARD_BREAKOUT:
            targets = {
                '127.2%': resistance + 0.272 * box_height,
                '161.8%': resistance + 0.618 * box_height,
                '261.8%': resistance + 1.618 * box_height,
                '423.6%': resistance + 3.236 * box_height
            }
        else:  # DOWNWARD_BREAKOUT
            targets = {
                '127.2%': support - 0.272 * box_height,
                '161.8%': support - 0.618 * box_height,
                '261.8%': support - 1.618 * box_height,
                '423.6%': support - 3.236 * box_height
            }
            
        return targets
    
    def _calculate_risk_reward_ratio(self, current_price: float, reference_level: float,
                                   fibonacci_targets: Dict[str, float], 
                                   signal_type: SignalType) -> float:
        """计算风险收益比"""
        # 风险：回到参考位的距离
        risk = abs(current_price - reference_level)
        
        # 收益：到第一个斐波那契目标位的距离
        first_target = list(fibonacci_targets.values())[0]
        reward = abs(first_target - current_price)
        
        # 风险收益比
        if risk > 0:
            return reward / risk
        else:
            return float('inf')
    
    def validate_follow_up(self, signal: BreakoutSignal, 
                          price_data: np.array, 
                          volume_data: np.array,
                          start_idx: int, 
                          periods: int = None) -> BreakoutSignal:
        """
        后续验证：检查信号发出后的表现
        
        Args:
            signal: 原始信号
            price_data: 价格数据
            volume_data: 成交量数据
            start_idx: 信号发出时的索引
            periods: 验证周期数
            
        Returns:
            更新后的信号
        """
        if periods is None:
            periods = self.confirmation_periods
            
        end_idx = min(start_idx + periods, len(price_data) - 1)
        
        if end_idx <= start_idx:
            return signal
            
        # 获取后续价格走势
        follow_up_prices = price_data[start_idx:end_idx+1]
        signal_price = signal.price
        
        if signal.signal_type == SignalType.UPWARD_BREAKOUT:
            # 向上突破后续验证
            max_price = np.max(follow_up_prices)
            min_price = np.min(follow_up_prices)
            
            # 有利移动和不利移动
            max_favorable_move = (max_price - signal_price) / signal_price * 100
            max_adverse_move = (signal_price - min_price) / signal_price * 100
            
            # 后续确认：价格持续在突破位上方
            prices_above_breakout = np.sum(follow_up_prices > signal.box_resistance)
            follow_up_confirmed = prices_above_breakout >= periods * 0.6
            
        else:  # DOWNWARD_BREAKOUT
            # 向下突破后续验证
            max_price = np.max(follow_up_prices)
            min_price = np.min(follow_up_prices)
            
            # 有利移动和不利移动
            max_favorable_move = (signal_price - min_price) / signal_price * 100
            max_adverse_move = (max_price - signal_price) / signal_price * 100
            
            # 后续确认：价格持续在突破位下方
            prices_below_breakout = np.sum(follow_up_prices < signal.box_support)
            follow_up_confirmed = prices_below_breakout >= periods * 0.6
        
        # 更新信号
        signal.follow_up_confirmed = follow_up_confirmed
        signal.max_favorable_move = max_favorable_move
        signal.max_adverse_move = max_adverse_move
        
        return signal
    
    def batch_validate_signals(self, boxes: List[Dict], 
                             price_data: np.array,
                             volume_data: np.array, 
                             macd_data: Dict,
                             dates: List[pd.Timestamp]) -> List[BreakoutSignal]:
        """
        批量验证突破信号
        
        Args:
            boxes: 箱体列表
            price_data: 价格数据
            volume_data: 成交量数据
            macd_data: MACD数据
            dates: 日期数据
            
        Returns:
            验证后的信号列表
        """
        validated_signals = []
        
        for box in boxes:
            end_idx = min(box['end_idx'], len(price_data) - 1)
            
            # 检查箱体后的突破
            for i in range(end_idx + 1, min(end_idx + 20, len(price_data))):
                signal = self.validate_breakout_signal(
                    box, price_data, volume_data, macd_data, i, dates
                )
                
                if signal is not None:
                    # 进行后续验证
                    signal = self.validate_follow_up(
                        signal, price_data, volume_data, i
                    )
                    validated_signals.append(signal)
                    break  # 每个箱体只取第一个有效突破信号
        
        return validated_signals
    
    def generate_signal_report(self, signals: List[BreakoutSignal]) -> Dict:
        """生成信号验证报告"""
        if not signals:
            return {'total_signals': 0}
            
        total_signals = len(signals)
        
        # 按确认级别统计
        level_counts = {}
        for level in ConfirmationLevel:
            level_counts[level.value] = sum(1 for s in signals if s.confirmation_level == level)
        
        # 按信号类型统计
        upward_signals = [s for s in signals if s.signal_type == SignalType.UPWARD_BREAKOUT]
        downward_signals = [s for s in signals if s.signal_type == SignalType.DOWNWARD_BREAKOUT]
        
        # MACD确认统计
        macd_confirmed = sum(1 for s in signals if s.macd_confirmed)
        volume_confirmed = sum(1 for s in signals if s.volume_confirmed)
        double_confirmed = sum(1 for s in signals if s.macd_confirmed and s.volume_confirmed)
        
        # 后续验证统计
        follow_up_signals = [s for s in signals if s.follow_up_confirmed is not None]
        follow_up_confirmed = sum(1 for s in follow_up_signals if s.follow_up_confirmed)
        
        # 平均指标
        avg_signal_strength = np.mean([s.signal_strength for s in signals])
        avg_breakout_pct = np.mean([s.breakout_percentage for s in signals])
        avg_volume_ratio = np.mean([s.volume_ratio for s in signals])
        avg_risk_reward = np.mean([s.risk_reward_ratio for s in signals if s.risk_reward_ratio != float('inf')])
        
        # 最佳表现信号
        if follow_up_signals:
            best_favorable = max(follow_up_signals, key=lambda x: x.max_favorable_move or 0)
            worst_adverse = max(follow_up_signals, key=lambda x: x.max_adverse_move or 0)
        else:
            best_favorable = None
            worst_adverse = None
        
        report = {
            'total_signals': total_signals,
            'signal_distribution': {
                'upward_breakouts': len(upward_signals),
                'downward_breakouts': len(downward_signals)
            },
            'confirmation_levels': level_counts,
            'validation_statistics': {
                'macd_confirmed': macd_confirmed,
                'volume_confirmed': volume_confirmed,
                'double_confirmed': double_confirmed,
                'macd_confirmation_rate': macd_confirmed / total_signals * 100,
                'volume_confirmation_rate': volume_confirmed / total_signals * 100,
                'double_confirmation_rate': double_confirmed / total_signals * 100
            },
            'follow_up_validation': {
                'total_follow_up': len(follow_up_signals),
                'confirmed_follow_up': follow_up_confirmed,
                'follow_up_success_rate': follow_up_confirmed / len(follow_up_signals) * 100 if follow_up_signals else 0
            },
            'average_metrics': {
                'signal_strength': avg_signal_strength,
                'breakout_percentage': avg_breakout_pct,
                'volume_ratio': avg_volume_ratio,
                'risk_reward_ratio': avg_risk_reward
            },
            'performance_highlights': {
                'best_favorable_move': best_favorable.max_favorable_move if best_favorable else None,
                'worst_adverse_move': worst_adverse.max_adverse_move if worst_adverse else None,
                'best_signal_id': best_favorable.signal_id if best_favorable else None,
                'worst_signal_id': worst_adverse.signal_id if worst_adverse else None
            }
        }
        
        return report


def main():
    """测试信号验证器"""
    print("信号验证器测试...")
    
    # 创建验证器
    validator = SignalValidator(
        macd_threshold=0.01,
        volume_ratio_threshold=1.5,
        breakout_threshold=0.5,
        confirmation_periods=3
    )
    
    print("信号验证器创建成功！")
    print(f"MACD阈值: {validator.macd_threshold}")
    print(f"量比阈值: {validator.volume_ratio_threshold}")
    print(f"突破阈值: {validator.breakout_threshold}%")
    print(f"确认周期: {validator.confirmation_periods}")

if __name__ == "__main__":
    main()