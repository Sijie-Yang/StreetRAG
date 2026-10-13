# 文献核查：地点文本的检索单元（2025–2026）

2026-10-12。目的：确认"把地点锚定文本以街道段为单元索引、沿路网按尺度扩展检索，并与 POI / 网格单元对照"这件事有没有人做过。

## 结论

1. **没有人用街道段做地点文本的检索单元。** 所有带文本的空间 RAG 都以 POI 为单元。
2. **没有人系统比较过检索单元（POI / 网格 / 街道）对 agent 回答和证据定位的影响。** "MAUP 在 RAG 里的翻版"这个说法在检索到的文献里没出现过。
3. **沿路网检索已经有人做了**（Perez 2026 的 network catchment；WalkRAG 的路径段），但一个用于数值指标、一个用于路径，都不用于文本，而且都没有和欧氏缓冲做对照。我们的对照实验会是新证据，但这两篇必须引，审稿人会问。
4. **主要风险来自 LifePlanner（2026-08）的发现**：agent 失败的主因是"证据获取不完整"，不是幻觉也不是检索结构。如果我们只测端到端答案，检索单元的效应可能被 agent 的探索行为淹没。实验必须把检索层单独评（固定查询下证据的相关性与定位准确性），端到端作为第二层。

## 逐篇

| 文献 | 时间 | 文本来源 | 检索单元 | 空间邻域 | 评估 | 与我们的关系 |
|---|---|---|---|---|---|---|
| [Spatial-RAG](https://arxiv.org/abs/2502.18470)（Yu, Mai 等） | 2025-02，后有 ACL Findings 版 | TourismQA（NYC 9,470 POI / 17,448 QA；Miami），TripAdvisor 问题 + 评论 | **POI 对象**（点/线/面） | LLM 生成空间 SQL：点、路径缓冲、区域 | P/R/F1/NDCG@k | 最接近的"POI+评论 RAG"；无单元对照；TourismQA 可作为外部基准 |
| [LifePlanner](https://arxiv.org/html/2608.25039) | 2026-08 | RedNote 约 20 万条帖子/评论，3,600+ 地点，10 km² | **POI**（"spatial anchor ID"） | 半径、沿路径缓冲（MCP 工具） | 667 条合成查询，Pass Rate、Place-set IoU 等 | 最强的 place-text agent 基准；结论"失败来自探索不足"是我们的设计风险 |
| [WalkRAG](https://arxiv.org/abs/2512.04790) | 2025-12 | 无地点文本（IR 语料是 TREC CAsT） | **路径段**（查询时生成） | GraphHopper 路网路由 + 路径缓冲 spatial join | 10 条路线 + 30 条问题，人工三档 | 用了路网段但不索引文本；评估很小 |
| [Network-based Spatial Context Retrieval](https://arxiv.org/html/2609.39437)（Perez） | 2026-09 | 无文本，开放地理数据算指标 | **路网可达集合（街道段 catchment）** | 沿路网步行距离 d | 1,434 条回答，faithfulness 基准 | 和"沿路网扩展"最像；作者明说没和欧氏缓冲对照 |
| [Real-time Spatial RAG, Madrid](https://arxiv.org/html/2505.02271v1) | 2025-05 | POI 描述（NGSI-LD） | POI | 用户画的矩形框 | 12 问 × 3 档 POI 数 | POI 单元；发现 650 个 POI 时准确率下降 |
| [AskNearby](https://arxiv.org/abs/2512.02502v1) | 2025-12 | "社区数据集"（未说明是否评论） | **15 分钟生活圈** | 图 + 向量 + 地理三层 RAG | 摘要无数字 | 区域单元的例子 |
| [SKG-grounded livability agents](https://arxiv.org/abs/2608.25952) | 2026-08 | 无文本 | 设施 / 住宅 / 路网 hub 实体 | Graph-RAG + 路由 | 原型演示，无量化 | 图单元，无文本 |
| [GeoRAG (2504.01458)](https://arxiv.org/abs/2504.01458) | 2025-04 | 3,267 篇地理文献 | 文档条目 | 无（按七个地理维度分类） | 未给数字 | 不是地点文本 |
| [Beyond the Townhall](https://arxiv.org/html/2604.16348v1) | 2026-03 | 无 | "spatial anchoring" 指 method of loci，不是索引 | — | 195 人实验 | 术语撞车，需在相关工作里区分 |
| [USTBench (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/a2fac827e992d55dcfdd4263e98528f4-Paper-Conference.pdf) | 2026 | 无文本（结构化 QA） | 道路实体 | — | 62k QA | 路网推理基准，非 RAG |
| [H3 hexagonal grids versus road segments](https://www.sciencedirect.com/science/article/pii/S2667091726000853) | 2026 | — | 六边形 vs 路段（骑行速度） | — | — | 非 RAG，但正好是"单元选择改变结果"的城市证据，可作论据 |

**没读到的**（网站拒绝或限流）：[Map-RAG](https://openreview.net/pdf?id=CUkjqutmHr)、[SpaRAGraph](https://dl.acm.org/doi/10.1145/3799424)、[Do We Still Need Gazetteers… Spatial Neuro-Symbolic Index](https://arxiv.org/html/2610.05028)（2026-10）。最后一篇标题里有"spatial index + chaining retrieval"，需要 Peter 自己看一眼，确认它没有做街道单元。

## 对论文设计的影响

- **Gap 成立，但要收窄表述**：不是"第一个空间 RAG"，而是"第一个把检索单元当作自变量来研究的地点文本 RAG"。相关工作里要老实承认 POI 单元（Spatial-RAG、LifePlanner）和路网邻域（Perez、WalkRAG）都已存在，我们的新东西是：(a) 街道段作为文本和数值共用的单元；(b) 单元与邻域的受控对照；(c) 路网尺度（句法半径）进入邻域定义。
- **评估分两层**：第一层固定查询、固定 agent 提示，只换索引，测检索到的证据的相关性和定位准确性（证据是否在对的街上）；第二层才是端到端回答。LifePlanner 的教训是只做第二层会测不出检索层的差异。
- **外部基准**：TourismQA（NYC，带评论）可以叠加 OSM 路网后复用，这样除了新加坡自有数据，还有一个别人认的基准。LifePlanner 是中文 RedNote，也可考虑，但要确认数据可得。
- **必须引用并区分**：Perez 2026（路网 catchment，无文本、无对照）、WalkRAG（路径段）、Spatial-RAG（POI+评论）、LifePlanner（POI 锚定帖子）、H3 vs road segments（单元效应的城市证据）。
