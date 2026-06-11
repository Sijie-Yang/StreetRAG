# SyntaxRAG 系统流程详解

## 核心创新：多特征融合的拓扑语义理解

SyntaxRAG 是一个通用的多特征检索增强生成系统，融合**空间句法特征**（多尺度整合度）和**街景感知参数**（19种感知分数），实现对城市街道的全面理解。

---

## 特征体系

### 1. 空间句法特征（Syntax Features）

**多尺度整合度**:
- `integration_R400`: 局部社区尺度
- `integration_R800`: 城市区域尺度
- `integration_R1200`: 城市整体尺度
- 可扩展：支持任意半径

**其他句法特征**:
- `connectivity`: 连接性
- `betweenness`: 中介中心性
- `closeness`: 接近中心性

### 2. 街景感知参数（Street Scene Affordance）

**19种感知分数** (示例):
- `thermal_affordance`: 热舒适度
- `visual_comfort`: 视觉舒适度
- `safety_perception`: 安全感
- `walkability`: 可步行性
- `green_view_index`: 绿视率
- `sky_view_factor`: 天空可视因子
- `noise_level`: 噪音水平
- ... (其他13种)

**特征维度**: 每个edge包含 3-5个尺度 × 整合度 + 19种感知参数 = **20+维特征空间**

---

## 系统架构

### 阶段 1: 多特征语义解析（Multi-Feature Semantic Parsing）

**输入**: 自然语言查询

**示例查询**:
- "找到高整合度且视觉舒适的主要街道"
- "热舒适度高但整合度低的步行路径"
- "安全和可步行性都好的区域"

**解析输出**: 多特征查询向量
```python
{
  "syntax_features": {
    "integration_R800": {"op": ">", "value": "high"},
    "scale": "auto"  # 自动选择或指定
  },
  "affordance_features": {
    "visual_comfort": {"op": ">", "value": 0.7},
    "thermal_affordance": {"op": ">", "value": 0.6}
  },
  "semantic_intent": "recommendation",
  "feature_weights": {  # 可学习或指定
    "syntax": 0.6,
    "affordance": 0.4
  }
}
```

---

### 阶段 2: 多特征检索（Multi-Feature Retrieval）

#### 2.1 特征索引结构

**Edge特征向量**:
```python
edge_features = {
  # 空间句法特征
  "integration_R400": float,
  "integration_R800": float,
  "integration_R1200": float,
  
  # 街景感知参数
  "thermal_affordance": float,
  "visual_comfort": float,
  "safety_perception": float,
  "walkability": float,
  ...  # 其他15种感知参数
  
  # 元数据
  "length": float,
  "geometry": geometry
}
```

#### 2.2 多维度检索策略

**检索方式**:

1. **语法特征检索** (Syntax Feature Retrieval):
   ```python
   WHERE integration_R800 > threshold_syntax
   ```

2. **感知特征检索** (Affordance Feature Retrieval):
   ```python
   WHERE visual_comfort > 0.7 AND thermal_affordance > 0.6
   ```

3. **混合检索** (Hybrid Retrieval):
   ```python
   WHERE integration_R800 > threshold 
     AND visual_comfort > 0.7
     AND safety_perception > 0.8
   ```

#### 2.3 特征权重自适应

**自动权重调整**:
- 根据查询语义自动调整语法特征和感知特征的权重
- 例如："高整合度" → syntax_weight = 0.8
- 例如："舒适的步行路径" → affordance_weight = 0.7

---

### 阶段 3: 多特征融合评分（Multi-Feature Fusion Scoring）

#### 3.1 特征维度评分

**语法维度评分** (`f_syntax`):
```python
f_syntax = weighted_score([
  integration_R400,
  integration_R800, 
  integration_R1200,
  connectivity,
  ...
], weights_syntax)
```

**感知维度评分** (`f_affordance`):
```python
f_affordance = weighted_score([
  thermal_affordance,
  visual_comfort,
  safety_perception,
  walkability,
  ...  # 其他感知参数
], weights_affordance)
```

#### 3.2 多维度融合

**融合策略**:
```python
# 基础融合
f_total = α × f_syntax + (1-α) × f_affordance

# 或者：每个特征维度单独评分后加权
f_total = Σ (w_i × f_i)
  where f_i ∈ {f_syntax_R400, f_syntax_R800, ..., f_thermal, f_visual, ...}
```

#### 3.3 语义相关性评分

**LLM语义匹配** (`f_semantic`):
- 基于查询语义和候选特征描述
- 评估候选与查询意图的匹配度

---

### 阶段 4: 多目标优化（Multi-Objective Optimization）

#### 4.1 多维特征空间

**候选表示**:
- 每个候选是一个多维特征向量
- 维度包括：多尺度整合度 + 19种感知参数 + 语义相关性

#### 4.2 Pareto Frontier 在多维空间

**目标空间**:
- 语法重要性 (多维度聚合)
- 感知舒适度 (多维度聚合)
- 语义相关性

**Pareto优化**:
- 在多维特征空间中识别最优候选
- 平衡不同特征维度的需求

---

### 阶段 5: 多特征解释生成（Multi-Feature Explanation Generation）

**LLM生成器**:
- 综合多尺度整合度和感知参数生成解释
- 解释候选在多个特征维度上的表现

**生成示例**:
```
推荐: Edge 123

解释:
- **拓扑重要性**: R800整合度 4.61e-7，表明从城市区域尺度看，
  这条街道具有良好的拓扑可达性，是连接不同区域的重要路径。

- **感知舒适度**: 
  - 视觉舒适度 0.75：街道景观良好，视野开阔
  - 热舒适度 0.68：有足够的遮阴，温度适宜
  - 安全感 0.82：环境安全，适合步行

- **综合评估**: 结合拓扑重要性（预测人流）和感知舒适度，
  这条街道既具有良好的可达性，又提供了舒适的步行环境，
  非常适合作为主要步行路径。
```

---

## SyntaxRAG 能处理的问题类型

### 类型 1: 语法特征查询

**问题**: "找到R800尺度下整合度最高的街道"

**处理**: 基于语法特征检索

### 类型 2: 感知特征查询

**问题**: "视觉舒适度高的步行路径"

**处理**: 基于感知参数检索

### 类型 3: 多特征组合查询

**问题**: "高整合度且视觉舒适的主要街道"

**处理**: 多维度检索 + 融合评分

### 类型 4: 特征权衡查询

**问题**: "整合度高但热舒适度也好的街道"

**处理**: 多目标优化，平衡不同特征

### 类型 5: 语义到特征的映射

**问题**: "舒适的步行路径"

**处理**: 
- 语义 → 特征映射：舒适 = visual_comfort + thermal_affordance
- 自动选择相关感知参数

### 类型 6: 多特征解释查询

**问题**: "为什么这条街道适合步行？"

**处理**: 
- 分析多维度特征
- 生成综合解释（语法 + 感知）

---

## 特征融合策略

### 策略 1: 加权融合

```python
score = w1 × f_syntax + w2 × f_affordance + w3 × f_semantic
```

### 策略 2: 特征选择

根据查询自动选择相关特征维度：
- "高整合度" → 只使用语法特征
- "舒适路径" → 主要使用感知特征
- "理想街道" → 使用所有特征

### 策略 3: 分层融合

```python
# 第一层：语法特征内部融合
f_syntax = fuse([integration_R400, R800, R1200])

# 第二层：感知特征内部融合  
f_affordance = fuse([thermal, visual, safety, ...])

# 第三层：跨特征类型融合
f_total = fuse([f_syntax, f_affordance, f_semantic])
```

---

## 技术实现要点

### 1. 特征标准化

**必要性**: 不同特征有不同的数值范围
- 整合度: ~1e-7 量级
- 感知参数: 0-1 范围

**标准化方法**:
- Min-Max标准化
- Z-score标准化
- 特征特定的标准化策略

### 2. 特征索引优化

**索引策略**:
- 语法特征：按不同尺度分别索引
- 感知特征：按特征类型索引
- 组合索引：支持多维度查询

### 3. 特征权重学习

**可学习权重**:
- 根据查询历史自动调整
- 用户偏好学习
- 任务特定的权重

---

## 与 SpatialRAG 的根本区别

| 维度 | SpatialRAG | SyntaxRAG |
|------|-----------|-----------|
| **特征类型** | 单一（地理坐标） | 多类型（语法+感知） |
| **特征维度** | 2-3维（x,y,z） | 20+维（多尺度+多感知） |
| **检索方式** | 空间查询 | 多维度特征检索 |
| **评分机制** | 空间接近度 | 多特征融合评分 |
| **查询能力** | 位置相关 | 多维度语义理解 |

---

## 系统优势

1. **通用性**: 可扩展支持任意特征类型
2. **多维度**: 同时考虑拓扑和感知特征
3. **灵活性**: 支持不同特征组合和权重
4. **解释性**: 多维度综合解释
5. **可扩展**: 容易添加新的特征类型
