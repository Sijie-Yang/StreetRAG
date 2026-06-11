# SyntaxRAG 系统架构

## 核心理念：通用多特征融合的语义理解

SyntaxRAG 是一个**通用的多特征检索增强生成系统**，支持任意类型和数量的特征。核心特征体系包括：

1. **多尺度空间句法特征**（Multi-Scale Space Syntax Features）
   - 整合度（Integration）：R400, R800, R1200, ... （任意半径）
   - 其他句法指标：connectivity, betweenness, closeness

2. **街景感知参数**（Street Scene Affordance Parameters）
   - 19种感知分数：thermal_affordance, visual_comfort, safety_perception, ...

---

## 通用特征架构

### 特征向量定义

每个 Edge 包含一个**多维特征向量**：

```python
edge_features = {
  # === 空间句法特征组 ===
  "syntax": {
    "integration_R400": float,    # 局部社区尺度
    "integration_R800": float,    # 城市区域尺度
    "integration_R1200": float,   # 城市整体尺度
    # ... 可扩展支持任意半径
    
    "connectivity": float,
    "betweenness": float,
    "closeness": float,
    # ... 其他句法特征
  },
  
  # === 街景感知特征组 ===
  "affordance": {
    "thermal_affordance": float,      # 热舒适度
    "visual_comfort": float,          # 视觉舒适度
    "safety_perception": float,       # 安全感
    "walkability": float,             # 可步行性
    "green_view_index": float,        # 绿视率
    "sky_view_factor": float,         # 天空可视因子
    "noise_level": float,             # 噪音水平
    # ... 其他12种感知参数
  },
  
  # === 元数据 ===
  "metadata": {
    "length": float,
    "geometry": geometry,
    "id": int
  }
}
```

**特征维度**: 动态可扩展
- 语法特征维度：N（N个不同半径的整合度 + 其他句法特征）
- 感知特征维度：19（或其他数量）
- **总维度**: N + 19 + 元数据维度

---

## 系统架构（通用设计）

### 阶段 1: 多特征语义解析（Multi-Feature Semantic Parsing）

**核心**: 将自然语言映射到多维度特征空间

**解析流程**:

```
用户查询: "找到高整合度且视觉舒适的主要街道"
  ↓
特征类型识别
  ├─→ 识别语法特征需求: "高整合度" → integration
  ├─→ 识别感知特征需求: "视觉舒适" → visual_comfort
  └─→ 识别语义意图: "主要街道" → recommendation
  ↓
多特征查询向量
{
  "syntax": {
    "integration_R800": {"op": ">", "threshold": "high"},
    "scale": "auto"  # 或指定R400/R800/R1200
  },
  "affordance": {
    "visual_comfort": {"op": ">", "threshold": 0.7}
  },
  "feature_weights": {
    "syntax": 0.6,
    "affordance": 0.4
  }
}
```

**特点**:
- **特征类型无关**: 可以识别任意特征类型
- **自动特征选择**: 根据查询自动选择相关特征
- **权重自适应**: 根据查询语义自动调整特征权重

---

### 阶段 2: 多特征检索（Multi-Feature Retrieval）

#### 2.1 通用特征索引

**索引结构**:
```
Edge Index:
  - 语法特征索引: {R400, R800, R1200, ...} → B-tree索引
  - 感知特征索引: {thermal, visual, safety, ...} → B-tree索引
  - 组合索引: (syntax_feature, affordance_feature) → 复合索引
```

#### 2.2 多维度检索策略

**检索方式**:

1. **单特征检索**:
   ```python
   # 只查询语法特征
   WHERE integration_R800 > threshold
   
   # 只查询感知特征
   WHERE visual_comfort > 0.7
   ```

2. **多特征AND检索**:
   ```python
   WHERE integration_R800 > threshold_syntax
     AND visual_comfort > 0.7
     AND thermal_affordance > 0.6
   ```

3. **多特征OR检索**:
   ```python
   WHERE integration_R800 > threshold_syntax
      OR visual_comfort > 0.8  # 任一满足即可
   ```

4. **特征组合检索**:
   ```python
   # 组合多个语法特征
   WHERE (integration_R400 > t1 OR integration_R800 > t2)
     AND visual_comfort > 0.7
   ```

#### 2.3 特征权重应用

**检索时考虑权重**:
```python
# 加权检索：优先考虑高权重特征
score = w1 × syntax_score + w2 × affordance_score
ORDER BY score DESC
```

---

### 阶段 3: 多特征融合评分（Multi-Feature Fusion Scoring）

#### 3.1 特征组内部评分

**语法特征组评分** (`f_syntax`):
```python
# 多尺度整合度融合
f_syntax = weighted_combine([
  integration_R400,
  integration_R800,
  integration_R1200,
  ...
], weights=[w_R400, w_R800, w_R1200, ...])

# 或：选择最相关的尺度
f_syntax = max_relevant_scale(integration_R{auto})
```

**感知特征组评分** (`f_affordance`):
```python
# 多感知参数融合
f_affordance = weighted_combine([
  thermal_affordance,
  visual_comfort,
  safety_perception,
  walkability,
  ...  # 其他感知参数
], weights=[w_thermal, w_visual, w_safety, ...])
```

#### 3.2 跨特征组融合

**融合策略**:

```python
# 策略1: 加权求和
f_total = α × f_syntax + (1-α) × f_affordance

# 策略2: 乘积融合（要求所有特征都满足）
f_total = f_syntax × f_affordance

# 策略3: 最大值融合（任一特征满足即可）
f_total = max(f_syntax, f_affordance)

# 策略4: 加权几何平均
f_total = (f_syntax^α) × (f_affordance^(1-α))
```

#### 3.3 语义相关性评分

**LLM语义匹配** (`f_semantic`):
- 基于查询语义和候选的多维度特征
- 评估候选与查询意图的整体匹配度

---

### 阶段 4: 多目标优化（Multi-Objective Optimization）

#### 4.1 多维特征空间表示

**候选表示**:
- 每个候选是一个**N+19维**特征向量
- 维度包括：
  - 语法特征：`[integration_R400, R800, R1200, connectivity, ...]`
  - 感知特征：`[thermal, visual, safety, walkability, ...]`
  - 语义相关性：`[semantic_score]`

#### 4.2 多目标Pareto优化

**目标空间**:
- 目标1: 语法重要性（多尺度整合度的聚合）
- 目标2: 感知舒适度（多感知参数的聚合）
- 目标3: 语义相关性
- （可扩展：支持更多目标）

**Pareto Frontier**:
- 在多维目标空间中识别最优候选
- 平衡不同特征维度的需求

---

### 阶段 5: 多特征解释生成（Multi-Feature Explanation Generation）

**LLM生成器**:
- 综合**所有相关特征**生成解释
- 解释候选在每个特征维度上的表现
- 说明为什么这些特征组合使得候选适合查询

**生成示例**:
```
推荐: Edge 123

多维度特征分析:
- **拓扑重要性（语法特征）**:
  * R400整合度: 2.1e-7（局部社区尺度，连接性好）
  * R800整合度: 4.6e-7（城市区域尺度，拓扑可达性高）
  * R1200整合度: 5.2e-7（城市整体尺度，是城市结构主干）
  
- **感知舒适度（感知特征）**:
  * 视觉舒适度: 0.75（街道景观良好，视野开阔）
  * 热舒适度: 0.68（有遮阴，温度适宜）
  * 安全感: 0.82（环境安全，适合步行）
  * 可步行性: 0.79（人行道良好，适合步行）

- **综合评估**:
  结合拓扑重要性（预测高人流）和多维度感知舒适度，
  这条街道在可达性和体验质量上都表现出色，是理想的
  主要步行路径。高整合度保证了可达性，而良好的感知
  参数确保了步行体验的舒适性。
```

---

## 可扩展性设计

### 1. 特征类型扩展

**添加新特征类型**:
```python
# 例如：添加交通特征
edge_features["traffic"] = {
  "traffic_volume": float,
  "congestion_level": float,
  ...
}

# 系统自动支持新特征类型的检索和评分
```

### 2. 特征数量扩展

- **语法特征**: 支持任意数量的尺度（R400, R500, R600, ...）
- **感知特征**: 当前19种，可扩展到任意数量
- **其他特征组**: 可添加新的特征组（traffic, environment, ...）

### 3. 检索策略扩展

- 支持任意特征组合
- 支持复杂的查询逻辑（AND, OR, NOT）
- 支持特征间的相关性约束

---

## 特征标准化与归一化

### 必要性

不同特征有不同的数值范围：
- 整合度: ~1e-7 量级
- 感知参数: 0-1 范围
- 长度: 米单位

### 标准化策略

```python
# 按特征类型标准化
normalized_features = {
  "syntax": normalize_syntax(integration_R800),  # 保留相对关系
  "affordance": normalize_affordance(visual_comfort),  # 0-1标准化
  ...
}
```

---

## 查询类型矩阵

| 查询类型 | 涉及特征 | 处理策略 |
|---------|---------|---------|
| **纯语法查询** | 只使用整合度 | 语法特征检索 |
| **纯感知查询** | 只使用感知参数 | 感知特征检索 |
| **语法+感知** | 两类特征 | 多特征融合 |
| **多尺度语法** | 多个半径整合度 | 多尺度组合 |
| **多感知组合** | 多个感知参数 | 感知特征融合 |
| **语义查询** | 自动特征选择 | LLM特征映射 |

---

## 实现要点

### 1. 特征索引

```python
# 为每种特征类型建立索引
indices = {
  "syntax": {
    "R400": BTreeIndex(integration_R400),
    "R800": BTreeIndex(integration_R800),
    ...
  },
  "affordance": {
    "visual": BTreeIndex(visual_comfort),
    "thermal": BTreeIndex(thermal_affordance),
    ...
  }
}
```

### 2. 特征权重

```python
# 可学习或用户指定
weights = {
  "syntax": 0.6,
  "affordance": 0.4,
  "syntax_scale": {"R400": 0.2, "R800": 0.5, "R1200": 0.3},
  "affordance_params": {"visual": 0.3, "thermal": 0.2, ...}
}
```

### 3. 特征扩展接口

```python
# 添加新特征的接口
def add_feature_group(name, features):
    """添加新的特征组"""
    edge_features[name] = features
    update_indices(name, features)
```

---

## 系统优势

1. **通用性**: 支持任意类型和数量的特征
2. **可扩展性**: 易于添加新特征类型
3. **灵活性**: 支持多种特征组合和权重策略
4. **多维度**: 同时考虑语法和感知特征
5. **解释性**: 多维度综合解释
