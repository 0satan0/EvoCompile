# 证据接到故事：框架图说明

源文件：

- **框架图**：[`evidence_to_story.dot`](evidence_to_story.dot) → [`evidence_to_story.png`](evidence_to_story.png)
- **检索图**：[`retrieve_scheme.dot`](retrieve_scheme.dot) → [`retrieve_scheme.png`](retrieve_scheme.png)
- **loop 图**：[`loop_evolve.dot`](loop_evolve.dot) → [`loop_evolve.png`](loop_evolve.png)
- **证据图**：[`evidence_channels.dot`](evidence_channels.dot) → [`evidence_channels.png`](evidence_channels.png)
- Agent I/O：[`agent_io.dot`](agent_io.dot) → [`agent_io.png`](agent_io.png)
- 模块 I/O 表：[`io_agent_table.md`](io_agent_table.md)

约定：实线 = 已进 `agent/loop.py` 闭环；虚线 = 工具已有、尚未作为 Retriever 检索键。**I 通道名字现已进检索**（`has_unfused_epilogue` → S6-fuse）。vanilla = `torch.compile(model, backend="inductor", fullgraph=False)`，**mode 缺省（default）**。

```mermaid
flowchart TB
  subgraph L0["L0  Case + runtime"]
    CASE["Training case<br/>HF / TIMM / TorchBench / Titan<br/>uncompiled nn.Module + inputs + optimizer"]
    RT["vanilla = torch.compile(model, backend=inductor, fullgraph=False)<br/>mode 缺省 = default"]
  end

  subgraph L1["L1  Evidence sensors  O = {C,G,D,A,I,P,H}<br/>raw FX / Triton 源码禁止进 LLM"]
    C["C source  IN loop<br/>forward() · named_children · LayerDrop"]
    G["G Dynamo graph / breaks  IN loop<br/>kinds: side_effect | data_dependent | vendor | setattr"]
    D["D guards / recompiles  dashed<br/>guards,recompiles · unique_graphs · cache_size_limit"]
    A["A AOT FX  dashed<br/>aten 直方图 + fp16/fp32 mix"]
    I["I Inductor  IN loop<br/>extern vs triton_* names → fusion_sites"]
    P["P profiler  IN loop<br/>GEMM/conv · attention · Adam · Triton · copy"]
    H["H runtime  IN loop<br/>eager_ms · vanilla_ms · peak GiB · traceback"]
  end

  subgraph L2["L2  Compress  eval/filter_feedback.py"]
    FILT["hints[]<br/>dtype_mix · data_dependent_python · mixed_triton_extern<br/>already_hardware_efficient · adam_launch<br/>recompile_cache_size_limit · many_static_shape_guards"]
  end

  subgraph L3["L3  Observation + compact fingerprint"]
    OBS["model suite amp opt · features{} · eager/vanilla Timing · model_tree · forward_src"]
    FP["compact_fp ~25 fields — similarity / evolve / LLM 只看这个"]
  end

  subgraph L4["L4  九个模块 + treememory"]
    COL["Collector  IN: case  OUT: Observation + I dump"]
    subgraph RETV["Retriever 无 LLM · walk treememory · Repairer 不重走"]
      KEYS["compact_fp<br/>H G kinds C P + I has_unfused_epilogue"]
      TREE["first-child-wins when DSL"]
      FAM["S1 Layer B patch  S2 LayerDrop<br/>S3* Adam leftover  S4 只编静态区<br/>S5 leave vendor  S6-fuse Layer C Triton<br/>F2-eager skip  F2-cnn/skinny/F6/identity"]
    end
    SIM["Similarity 双通道 kNN"]
    OPT["Optimizer  OUT: recipe JSON"]
    RW["RewriteWriter  S1-generic → eval/rewrites/generated"]
    WRT["TritonWriter  sites+contract → eval/kernels[/generated]"]
    EVA["Evaluator  每轮 _load 未编译 · in-memory patch + compile vs 同一次 vanilla"]
    REP["Repair  region crash；rewrite→RW；Triton→TW"]
    EVO["Evolve  trial · consider · explore_from · promote"]
    MEM["treememory JSON"]
  end

  subgraph L5["L5  Recipe 三层  (不是旧动作族 A–E)"]
    LA["A compile region  无新 .py"]
    LB["B method patch  apply(model)"]
    LC["C generated kernel"]
    SK["always recipe JSON；B/C 仅 scheme 要时"]
  end

  subgraph L6["L6  测量 + 论文"]
    M["Metrics  beat_naive: agent &lt; 0.95×vanilla"]
    GRID["网格搜索 BASELINE  对照不是方法"]
    STORY["1 Guards D  2 FX A/G  3 Fusion I  4 Custom I/P  5 Adam P"]
  end

  DIAG["Diagnosis: break / recompile / region / dynamic / fusion / vendor / launch / already efficient"]

  CASE --> COL
  RT --> COL
  COL --> C & G & P & H & I
  C & G & P & H & I --> FILT
  D -.-> FILT
  A -.-> FILT
  COL --> OBS
  OBS --> FP
  FILT -.-> OBS
  FP --> KEYS
  MEM -.-> TREE
  KEYS --> TREE
  TREE --> FAM
  FAM -->|hard SchemeHit| SIM
  MEM -.->|trials win/lose| SIM
  SIM -->|chosen + cards| OPT
  OBS --> OPT
  OPT --> LA & LB & LC & SK
  OPT -->|S1-generic| RW
  OPT -->|S6| WRT
  RW --> LB
  WRT --> LC
  LA & LB & LC & SK --> EVA
  EVA --> M
  EVA --> DIAG
  DIAG --> STORY
  M --> STORY
  EVA -->|rewrite crash| RW
  EVA -->|Triton crash| WRT
  EVA -->|region crash| REP
  REP -->|shrink region| LA
  EVA -->|every trial| EVO
  EVO -->|explore| SIM
  EVO -.->|consider/promote/path| MEM
  M -.->|ran but not +5%| OPT
  GRID -.-> STORY
```

## 怎么读

1. **L1 不要把 raw dump 画成 agent 输入。** C/G/P/H/**I 名字**已经进 Collector；D/A 仍虚线（D 只给 recipe 上的 `mark_dynamic`，不是 walk 谓词）。Triton 源码不进 LLM。
2. **L4 九个模块。** Retriever 仍选方案名；S1 走 Layer B（catalog / `allow_logging` / RewriteWriter）；S6-fuse 是 Layer C「compile/厂商没覆盖的融合洞」，**不是**必须赢过 cuBLAS。rewrite 报错回 RewriteWriter；Triton 报错回 TritonWriter；其它 compile 报错走 Repair 一次再 explore。轨迹是本 run 的 `tried[]` / `paths`。
3. **L5 三层（论文 Layer A/B/C）。** 不要画成旧动作族 A–E。A = 编译区域（含 skip / leave vendor / `mark_dynamic`），不写新 `.py`；B = 内存方法补丁，永不覆盖 disk `model.py`；C = 生成 Triton。每轮 recipe 不叠：Evaluator `_load` 新的未编译模型，对照仍是 collect 那一次 vanilla。
4. **L6 网格搜索用虚线**（negative control）。
5. **每条 paper story 必须指回 L1 某一格，并用 Layer A/B/C 写动作。** Fusion 故事是 generate Triton，不只是「停手别改 Flash」。

```bash
cd ${GKO}
for f in evidence_to_story retrieve_scheme loop_evolve agent_io evidence_channels; do
  dot -Tsvg "graph/${f}.dot" -o "graph/${f}.svg"
  dot -Tpng -Gdpi=160 "graph/${f}.dot" -o "graph/${f}.png"
done
```
