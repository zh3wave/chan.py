# ZigZag策略快速开始指南

## 🚀 一键运行

### 主要分析器（推荐）
```bash
cd BoxBreakout
python zigzag_visual_analyzer.py
```
**输出**: `zigzag_detailed_analysis_customer_recommended.png`

### 增强版可视化
```bash
python enhanced_zigzag_visualizer.py
```
**输出**: `enhanced_zigzag_swing_points_详细标注.png`

## 📊 核心文件

| 文件 | 功能 | 重要性 |
|------|------|--------|
| `okx_zigzag_standard.py` | 核心ZigZag算法 | ⭐⭐⭐⭐⭐ |
| `zigzag_visual_analyzer.py` | 主要可视化分析器 | ⭐⭐⭐⭐⭐ |
| `test_fixed_zigzag.py` | 算法测试验证 | ⭐⭐⭐⭐ |
| `eth_zigzag_backtest.py` | 回测系统 | ⭐⭐⭐ |

## 🔧 关键参数

### 用户推荐配置（高敏感度）
```python
deviation = 1.0  # 1%价格变化阈值
depth = 10       # 最小间隔深度
order = 2        # argrelextrema参数（已修复）
```

### 快速测试
```python
from okx_zigzag_standard import OKXZigZag

zigzag = OKXZigZag(deviation=1.0, depth=10)
swing_points, zigzag_line = zigzag.calculate(highs, lows)
print(f"识别到 {len(swing_points)} 个摆动点")
```

## 🎯 核心修复

1. **order参数**: `depth//2` (5) → `2` ✅
2. **过滤逻辑**: 保留更极端的同类型点 ✅
3. **验证结果**: 成功识别真正最高点 ✅

## 📈 预期结果

- **摆动点数量**: ~26个（ETH 5分钟数据）
- **支撑阻力位**: ~5个关键位置
- **图表输出**: 高质量可视化分析图

## 🔍 故障排除

- **摆动点太少**: 降低`deviation`参数
- **摆动点太多**: 增加`deviation`或`depth`参数
- **算法问题**: 运行`test_fixed_zigzag.py`验证

---
**快速回忆**: 这是一个基于价格摆动点的箱体突破策略，核心是修复后的ZigZag算法，能准确识别局部极值点并构建交易信号。