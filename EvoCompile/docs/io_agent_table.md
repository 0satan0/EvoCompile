# 图上应出现的输入 / 输出 / Agent 模块

对应 **框架** [`evidence_to_story.png`](evidence_to_story.png)、**检索** [`retrieve_scheme.png`](retrieve_scheme.png)、**loop** [`loop_evolve.png`](loop_evolve.png)、**证据** [`evidence_channels.png`](evidence_channels.png)、I/O [`agent_io.png`](agent_io.png)。说明：[`evidence_to_story.md`](evidence_to_story.md)。

约定：

- **画**：主框图里要有独立节点（或 cluster 内的盒子）。
- **旁注**：不必单独成盒，写在父节点或 caption 里。
- **虚线**：工具已有、尚未进入 `agent/loop.py` 检索键；论文 case study 仍建议画出。

成功判据（图 L6 必须写出来）：`agent_ms < 0.95 × vanilla_ms`，或 naive compile 失败而 agent 能跑。vanilla = **default** `torch.compile(model, backend=inductor, fullgraph=False)`，不是 reduce-overhead / max-autotune。

---

## 1. Agent 模块（图 L4，九个盒子 + treememory 文件夹）

| 图上名字 | 代码 | 输入（必须能从箭头读出） | 输出（必须能从箭头读出） | 画法 | 为什么要画 |
|---|---|---|---|---|---|
| **CollectorAgent** | `agent/agents/collector.py` | case 名、suite、AMP 开关、uncompiled 模型 | `Observation`：eager/vanilla Timing、features、model_tree、forward_src、可选 profiler | **实线主盒** | 证据入口。没有它，后面全是盲搜。 |
| **RetrieverAgent** | `agent/agents/retriever.py` + `memory/treememory` | compact fingerprint（H/G/C/P 谓词，**不是**模型名硬记、**不是** embedding） | `SchemeHit`：`name, node_id, reason, recipe_hint, do_not, path` source=hard | **实线主盒 + first-child-wins**，标注「无 LLM」 | 方案名不由 LLM 选。Repairer **不重走**这棵树。 |
| **SimilarityAgent** | `agent/agents/similarity.py` | fingerprint + hard hit + `trials` | 近邻赢/败卡、`banned[]`；仅 `identity-fallback` 可 transfer | **实线主盒** | 双通道：赢局迁移，败局禁止复探 accel。 |
| **OptimizerAgent** | `agent/agents/generator.py` + `agent/prompts.py` | Observation + SchemeHit + compact win/loss cards | recipe JSON（`eval/recipe.py` 能 apply）+ `llm_raw`；S1-generic 时 `apply_rewrite rewrite=__generate__`；S6 时 `replace_pattern kernel=__generate__` | **实线主盒** | 把方案落成可执行动作；prompt 不含整棵树。源码不内联进 JSON。 |
| **RewriteWriter** | `agent/agents/rewrite_writer.py` | break_sample + kinds（`side_effect \| data_dependent \| vendor \| setattr`）+ 截断 forward + rewrite traceback | `eval/rewrites/generated/<id>.py`；inject rewrite id 进 recipe | **实线主盒** | 未见过的热路径 break 没有 catalog kind。只写 `apply(model)`，不覆盖 disk `model.py`。不选方案名。 |
| **TritonWriter** | `agent/agents/triton_writer.py` | fusion sites（`fusion_sites.py`）+ 插入合同（`register()`、`torch.ops.gko`、FX pass）+ Triton traceback | `eval/kernels/generated/<id>.py`；inject kernel id 进 recipe | **实线主盒** | 自定义融合很泛，不能只用一个 `missed_gemm_epilogue` 旗标。不选方案名。 |
| **EvaluatorAgent** | `agent/agents/evaluator.py` | Observation（对照 eager/vanilla）+ recipe | `EvalResult`：ok/ms/compile_s/peak、applied、beat_naive、traceback | **实线主盒** | 唯一打分器。每轮 `_load` 未编译模型再 in-memory patch + compile，对照 **同一次** collect 的 vanilla（不叠上一轮 recipe）。 |
| **RepairAgent** | `agent/agents/repairer.py` | Observation 子集 + **失败 recipe** + **非 rewrite / 非 Triton traceback** | 新 recipe（drop fullgraph / 缩小 compile path）；每 scheme 一次 | **实线主盒，标注「不重新 retrieve」** | 修 Layer A 边界。rewrite 报错回 RewriteWriter；Triton 报错回 TritonWriter。二次 compile crash → explore。 |
| **EvolveAgent** | `agent/agents/evolver.py` | fingerprint + SchemeHit + EvalResult + tried/banned | trial 记录；`consider` 插叶；`explore_from` 下一方案；`promote` 窄叶 | **实线主盒** | 记忆可扩展：trials 追加、tree 可插、stop leaf 不 explore 进 S3。 |
| **Orchestrator** | `agent/loop.py` | CLI：`--case --max-rounds --no-llm --no-similarity --no-evolve` | `observation.json`、`retrieval.json`、`rounds/*.json`、`final_recipe.json`、`summary.json` | **cluster 外框或 caption** | collect→retrieve→sim→optimize→[RewriteWriter]→[TritonWriter]→eval；rewrite crash→RW；Triton crash→TW；其它 crash→repair。 |
| **Memory** | `memory/treememory`（tree / schemes / trials / cases / paths）；人读 `scheme_index` | 谓词树 + 实验日志 | Retriever 的 walk；Similarity 的 kNN；Evolve 的写回 | **文件夹，虚线进 Retriever/Similarity** | artifact 是 JSON recipe + 可选 B/C 文件，不是改 Dynamo harness、不是新 `model.py`。 |
| **LLMClient** | `agent/llm.py` | system/user prompt（compact cards only） | 文本，再 `parse_json_object` | **旁注在 Optimizer/Repair/RewriteWriter/TritonWriter/Evolve 上** | 可 `--no-llm`：S1-generic 仍写 taxonomy heuristic `.py`；S6 退化为 catalog `addmm_gelu`。 |

不要画成 agent 的东西：`filter_feedback.py`（压缩器，L2）、`diagnose_compile.py` / `inspect_features.py`（Collector 的传感器）、`apply_recipe`（执行器，L5）、网格搜索脚本（L6 对照）。

---

## 1b. Retriever 检索（主图 L4 内必须展开；详图 `retrieve_scheme`）

匹配方式：**treememory 有序 `when`，first-child-wins**。Similarity 是加权 ℓ1+Hamming kNN，**不是** LLM 选 `name`。Optimizer 只在命中方案内写 recipe JSON。

**IN 检索键（必须能从 Observation 读出）**

| 组 | 字段 | 来自 | 用来分什么 |
|---|---|---|---|
| H 失败/计时 | `eager.ok` `vanilla_ok` `eager_ms` `vanilla_ms` `ratio=eager/vanilla` | Collector 计时 | B-loader / S4 / F2-eager / S3 要不要 fullgraph |
| G 图 | `fwd_graphs` `fwd_breaks` `break_sample` `break_kinds`（side_effect / data_dependent / vendor / setattr） | `dynamo.explain` fwd | S1（热路径碎：logging / catalog / RewriteWriter）vs 0-break 走 S3/F2 |
| C 源/结构 | `layerdrop_n` `conv_param_frac` `linear_param_frac` `DenseLayer` `hidden` `n_layer` `vocab` `n_params` `n_tensors` `arch_class` | inspect + named_children | S2 / F6 / S3-tiny-vit / S3-embed / F2-skinny / F2-cnn |
| P+opt | `opt` 是否标准 Adam；profiler `other%`=sync；名字门：detectron / qat / opacus / HF_LLM / FNet / yolo / speech | Collector | S3* / S5 / S4 / S3-step / S3-fc |
| I 融合点 | `fusion_sites` / `has_unfused_epilogue`：compile 没融合或厂商核覆盖不到（GEMM+epi、split silu\|mul、多 distinct poi）。**不是**「必须赢过 cuBLAS」。裸 leftover GEMM 无 epi、Flash-only → 不是洞 | Collector dump_trace | **S6-fuse** |

**first-hit 顺序 → 方案类 → recipe_hint 层**

| 顺序 | 谓词（命中即停） | 方案 `name` | 典型 hint | 层 |
|---|---|---|---|---|
| 0 | `not eager.ok` | **B-loader** | 不 compile | — |
| 1 | `vanilla_ok is False` ∧ detectron/maskrcnn | **S4** | `disable_forward` + `compile path=backbone` | A |
| 1 | 同上 ∧ qat / quantized | **S5-qat** | identity fp32 GraphModule | A |
| 1 | 同上 ∧ opacus | **S5-opacus** | compile `_module`，hooks eager | A |
| 1 | 同上 else | **S4** | identity `fullgraph=False` | A |
| 2 | HF_LLM 名 ∩ Adam | **S3-deep** | `compile_optimizer`，评测 `--no-amp` | A |
| 3 | vanilla > 1.05×eager 且（步长&lt;15ms 或 conv≥0.3 且 &lt;40ms） | **F2-eager** | `actions: []` | A skip |
| 4 | breaks≥10，或 ≥5 且 ratio≤1.08 且 setattr/nonzero/Dynamic | **S1** | logging → `allow_logging`；YOLO/speech catalog；else RewriteWriter + `fullgraph` | B+A |
| 5 | `layerdrop_n ≥ 1` | **S2** | `hf_layerdrop_off` + fullgraph + `compile_optimizer` | B+A |
| 6 | DenseLayer ≥ 12 | **F6** | identity，禁 fullgraph | identity |
| 7 | CombinedOptimizer / EmbeddingBag | **S5-embed** | identity，不编 opt | A |
| 8 | FNet / GoogleFnet | **S3-step** | `compile_step` + `actions: []` | A |
| 9 | conv&lt;0.3 ∧ lin≥0.5 ∧ Adam ∧ sync≥40% ∧ n_params≥10 | **S3-fc** | compiled Adam（VGG 可 fullgraph） | A |
| 10 | conv≥0.3 ∧ 0-break；若 Adam 且 sync≥70% 且 vanilla_ms≥80 | **S3-fc** 否则 **F2-cnn** | compiled Adam / identity | A / identity |
| 11 | LSTM/RNN/cuDNN in break or profiler | **S5** 或 **S5+S3** | 不编 LSTM；长步长仍编 Adam | A |
| 12 | Adam ∧ 0-break（再拆） | 见下行 | | A |
| 12a | distil / n_layer≤6 且 ratio≥2.3 / hidden≤256 / n_tensors&gt;500 / 深层但 vanilla_ms&lt;80 | **F2-skinny** | identity | identity |
| 12b | ViT embed≤192 | **S3-tiny-vit** | fullgraph + Adam | A |
| 12c | vocab≥1e5 或 params≥200M | **S3-embed** | compiled Adam | A |
| 12d | n_layer≥20 且 vanilla_ms≥80 | **S3-deep** | compiled Adam | A |
| 12e | ratio≤1.20 ∧ hidden≥768 ∧ n_layer≥12 | **S3-unfused** | fullgraph + Adam | A |
| 12f | ratio 1.20–2.20 或 setattr/.item | **S3b** | **no** fullgraph + Adam | A |
| 12g | 其余 Adam leftover | **S3b** | 同上 | A |
| 13 | vanilla_ms&lt;5 或 n_params&lt;0.2 | **identity** | naive `compile(model)` | identity |
| 14 | `has_unfused_epilogue`（更早叶子未命中） | **S6-fuse** | `replace_pattern kernel=__generate__` | C |
| else | 无命中 | **identity** | 同上 | identity |

图上 **不要** 把方案画成「按模型名查表」。名字门（detectron / yolo / FNet）只是谓词里的快捷键，主路径仍是 breaks / layerdrop / conv_frac / Adam / ratio。

---

## 2. 输入：证据通道 O = {C,G,D,A,I,P,H}（图 L1）

| 记号 | 名称 | 采集命令 / API | 进闭环的形态 | 画法 | 论文故事 |
|---|---|---|---|---|---|
| **C** | Source | `inspect.getsource(forward)`、`named_children`、`.layerdrop` | `forward_src`、`model_tree`、`layerdrop_n` | **实线** | 指出 break 落在用户文件哪一行，而不是 torch 内部。 |
| **G** | Dynamo graph / breaks | `torch._dynamo.explain`；`TORCH_LOGS=graph_breaks`；scope=fwd/step | `fwd_graphs`、`fwd_breaks`、`break_sample` | **实线** | 「CSV 6–8 break 是 harness glue；fwd 才是模型碎没碎。」 |
| **D** | Guards / recompiles | `TORCH_LOGS=guards,recompiles`；`torch._dynamo.utils.counters` | 现有：`dynamo_counters`（弱）。完整 guard 种类 / `cache_size_limit` 在 filter 里 | **虚线盒** | seq_len `TENSOR_MATCH` → `mark_dynamic`；重编译次数下降。 |
| **A** | AOTAutograd FX | `TORCH_LOGS=aot_graphs,graph_code` | filter：aten 直方图、dtype mix。**不要**整图节点列表 | **虚线盒** | 1 张 fused graph vs 25 silent graphs；AMP 外层 fp16/fp32 混用。 |
| **I** | Inductor output | Collector `dump_trace`（默认） | `inductor_extern` / `inductor_triton` **名字** → `fusion_sites`。丢掉 Triton 源码 | **实线** | leftover GEMM+epi → S6 TritonWriter；裸 leftover cuBLAS 不算 site。 |
| **P** | Profiler | `torch.profiler` CUDA | `profiler.top_kernels` + 桶：gemm_conv / attention / adam / triton / copy | **实线** | Adam leftover → `compile_optimizer`；attention 桶 → 不要重写 Flash。 |
| **H** | Hardware / runtime | 计时循环、AMP、`max_memory_allocated` | `eager_ms`、`vanilla_ms`、`vanilla_ok`、`peak_gib`、traceback | **实线** | 对照永远是同进程 eager / default vanilla。 |

**不要画进主输入箭头的：** 完整 TREE_GUARD_MANAGER、每个 TENSOR_MATCH stride、整份 FX graph print、Triton kernel 源码、`<1%` 的 `aten::empty`。这些写在 L2 caption「dropped」。

---

## 3. 中间对象（图 L3、箭头标签）

| 对象 | 字段（图上写出最关键的） | 生产者 | 消费者 | 画法 |
|---|---|---|---|---|
| **Observation** | model, suite, amp, opt, features{}, eager, vanilla, model_tree, forward_src | Collector | Retriever, Similarity, Optimizer, Repairer, Evaluator, Evolve | **L3 中心盒** |
| **fingerprint** | ~25 字段 timings/breaks/layerdrop/conv/lin/sizes/Adam/booleans | `fingerprint.compact_fp` | Retriever walk、kNN、evolve LLM | **L3 旁盒** |
| **SchemeHit** | name（S1/S2/S3/S4/S5/F2…）, source=hard\|similarity\|explore, similar[], similar_losses[], banned[] | Retriever + Similarity（+ Evolve explore） | Optimizer, Evolve | **箭头标签**，可小盒 |
| **recipe JSON** | `actions[]` + `compile_optimizer` / `compile_step` + 可选 `apply_rewrite` / `replace_pattern` | Optimizer 或 Repairer 或 RewriteWriter/TritonWriter(inject) 或 hint | Evaluator → `apply_recipe` | **L5 cluster 的总输入** |
| **EvalResult** | ok, ms, beat_naive, reason, err, applied | Evaluator | Orchestrator；失败进 Repair；trial 日志；Evolve | **L6 Metrics 盒** |
| **trial / path** | verdict win/tie/lose/crash；tried 顺序 + best | loop → treememory | 下次 Similarity / explore 禁表 | **Memory 盒内** |

---

## 4. 输出：Recipe 三层（图 L5；论文 Layer A/B/C，不要画成旧动作族 A–E）

| 图上层 | 对应 op / 标志 | 典型 case | 网格搜索能否替代 | 画法 |
|---|---|---|---|---|
| **A  compile region** | `compile path=`、`compile_optimizer`、`compile_step`、`disable_forward`、`skip`、`mark_dynamic`、`allow_logging`、S5 leave vendor | HF Adam；FNet step；resnet18 skip；LSTM/Flash 不编 | 否（mode 网格不能选 region） | **实线，最大盒**；**不写** `.py` |
| **B  method patch** | catalog `rewrite_class_forward` / `hf_layerdrop_off` / speech；S1-generic `apply_rewrite` → RewriteWriter `apply(model)` | yolov3 setattr、print → `allow_logging`、Bart LayerDrop | 否 | **实线**；内存绑定，**永不覆盖** disk `model.py` |
| **C  generated kernel** | `replace_pattern` + TritonWriter；catalog 或 `eval/kernels/generated/` | Linear+GELU missed epilogue；其它 I leftover site | 否 | **实线**；不是新 `forward` |
| **always JSON** | 每轮都有 recipe；B/C 文件仅当 scheme 要 `__generate__` | — | mode/cudagraphs 网格**能**做，标辅助 | **旁注** |

---

## 5. 对照与论文出口（图 L6）

| 图上名字 | 是什么 | 画法 | 作用 |
|---|---|---|---|
| **Metrics** | eager / vanilla=default / agent；`gko_eval.csv` 冻结官方 vanilla | **实线** | 数字 |
| **Grid-search baseline** | 全表 `mode=default already frozen`；补 `reduce-overhead`、`max-autotune` | **虚线框，标注 negative control** | 证明涨点不是换 mode |
| **Diagnosis 菱形** | actionspace 那棵「time 在哪」树 | **菱形，从 Evaluator 指向 Story** | 把测量接到因果 |
| **Paper case studies** | 五条证据链（Guards / FX / Fusion / Custom kernel / Adam leftover） | **实线终点** | 专业故事；数字可以不大 |

---

## 6. 建议的图面清单（画的时候对着勾）

主图上 **必须出现**：

1. Case + vanilla=default  
2. C, G, P, H, **I**（实线）和 D, A（虚线）  
3. filter_feedback 与 dropped 列表  
4. Observation  
5. 九个模块 + treememory 文件夹（含 RewriteWriter）  
5b. Retriever 展开：检索键 H/G kinds/C/P/**I**、first-child-wins、方案类 S1–S6 / F2 / identity、OUT SchemeHit  
5c. Similarity 双通道 + RewriteWriter + TritonWriter + Evolve  
6. 箭头：SchemeHit、recipe JSON、rewrite traceback→RW、Triton traceback→TW、region traceback→Repair（不 retrieve）、eval→trial→explore  
7. L5 三层 A/B/C（compile region / method patch / generated kernel）；不要画旧动作族 A–E  
8. beat_naive 公式  
9. 网格搜索虚线对照  
10. 五条 paper story（动作用 Layer A/B/C）  

主图上 **不要出现**：

- 每个 inductor flag 的穷举表（那是 baseline 附录）  
- LLM 作为总控  
- 把 `torch.compile` 画成唯一动作  
- raw Triton / 完整 FX  
- Titan 8 卡细节（另图）；这里只留 case 入口  

若版面不够，拆成两张：**图 1 系统闭环（L0–L5）**，**图 2 证据→故事（L1 虚线通道 + Diagnosis + 五条 case study + 网格对照）**。
