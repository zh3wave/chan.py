# 版本历史管理

## 版本备份说明

本目录用于存储项目的不同版本，确保可以随时回退到之前的稳定版本。

## 版本列表

### V4.0 (当前版本) - 2025-09-30
**文件**: `zigzag_visual_analyzer_v4.0_清晰图表版_20250930_110605.py`

**特性**:
- ✅ 真正的时效性权重：完全放弃遥远价位（前40%历史数据）
- ✅ 清晰简洁的图表设计：按用户示例图样式绘制
- ✅ 优化箱体显示：最多5个箱体，按时间顺序，最新箱体绿色标识
- ✅ 简化颜色方案：淡色背景，优化ZigZag摆动点颜色
- ✅ 移除冗余信息：简化标注，突出关键价格区间

**改进要点**:
1. 支撑阻力位识别限制在最近60%数据点
2. 箱体识别按时效性评分，越新越重要
3. 图表信息密度大幅降低，视觉清晰度显著提升
4. 颜色方案优化，符合短线交易需求

**测试结果**:
- 图表清晰度: 显著提升
- 信息密度: 大幅降低
- 用户体验: 符合示例图要求
- 时效性权重: 真正实现放弃遥远价位

### V3.1 - 2025-09-26
**文件**: `enhanced_analyzer_advanced_v3.1_20250926_110803.py`

**特性**:
- ✅ 实现箱体重叠检测和聚合功能
- ✅ 重叠箱体使用Simple边界（ZigZag点）
- ✅ 混合算法：Enhanced优先，Simple补充
- ✅ 智能冲突处理和强度增强
- ✅ 统一版本命名规范（V1.0, V1.1格式）
- ✅ 完整的突破信号分析和斐波那契目标位

**测试结果**:
- 测试标的: sz.000063
- 箱体数量: 8个（聚合前10个）
- Enhanced识别: 5个箱体
- Simple补充: 1个箱体
- 聚合箱体: 2个
- 突破信号: 8个
- 聚合效果: 成功减少重叠

### v2.0 - 2025-09-25
**文件**: `integrated_test_v2.0.py`

**特性**:
- ✅ 简化为2分图布局（主图 + 成交量/量比图）
- ✅ 移除MACD依赖，专注核心逻辑
- ✅ 保留完整的箱体识别和突破检测
- ✅ 优化成交量和量比分析
- ✅ 清晰的信号标记和结果输出

**测试结果**:
- 测试标的: sz.000063
- MACD确认率: 已移除
- 成交量确认率: 良好
- 后续验证: 待优化

## 版本回退操作

### 方法1: 手动回退
```powershell
# 回退到v2.0版本
Copy-Item versions/integrated_test_v2.0.py integrated_test.py
```

### 方法2: Git版本控制
```powershell
# 查看提交历史
git log --oneline

# 回退到特定提交
git checkout <commit_hash> -- integrated_test.py

# 或者创建新分支进行实验
git checkout -b experiment_v3
```

### 方法3: 创建版本标签
```powershell
# 为当前版本打标签
git tag -a v2.0 -m "简化版本，移除MACD依赖"

# 回退到标签版本
git checkout v2.0 -- integrated_test.py
```

## 升级前的准备工作

在升级脚本之前，请务必：

1. **备份当前版本**
   ```powershell
   Copy-Item integrated_test.py versions/integrated_test_v$(Get-Date -Format "yyyyMMdd_HHmmss").py
   ```

2. **提交到Git**
   ```powershell
   git add .
   git commit -m "备份v2.0版本，准备升级"
   ```

3. **创建实验分支**
   ```powershell
   git checkout -b strategy_a_upgrade
   ```

## 版本比较

如需比较不同版本的差异：
```powershell
# 比较当前版本与v2.0的差异
git diff HEAD versions/integrated_test_v2.0.py

# 或使用文本比较工具
code --diff integrated_test.py versions/integrated_test_v2.0.py
```

## 注意事项

- 每次重大修改前都要创建版本备份
- 保持版本命名的一致性
- 记录每个版本的主要特性和测试结果
- 定期清理过旧的版本文件