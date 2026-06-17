# Traffic Agent Evaluation Harness Run Report

- **Run Timestamp**: 2026-06-18 00:14:00
- **Success Rate**: 100.0% (4/4 Passed)
- **Average Execution Latency**: 28.03s

## Summary Table

| Test Case | Category | Expected Intent | Actual Intent | Image Gen? | Latency | Status | Notes / Failures |
|---|---|---|---|---|---|---|---|
| TC001 | Simulation Diagnostic | `DIAGNOSTIC` | `DIAGNOSTIC` | True | 33.2s | ✅ | N/A |
| TC002 | Academic Query (Basic) | `QUERY_ONLY` | `QUERY_ONLY` | False | 21.1s | ✅ | N/A |
| TC003 | Academic Query (LaTeX & PyTorch) | `QUERY_ONLY` | `QUERY_ONLY` | False | 31.4s | ✅ | N/A |
| TC004 | Multi-Seed Diagnostic | `DIAGNOSTIC` | `DIAGNOSTIC` | True | 26.4s | ✅ | N/A |

## Detailed Test Case Output Previews

### [TC001] Category: Simulation Diagnostic
- **User Prompt**: *"请帮我诊断最新的强化学习控制日志并画出收敛图。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  # 自动化仿真评估报告：基于强化学习的交通信号控制收敛性诊断

## 摘要

本报告针对基于强化学习（RL）的交通信号控制模型在 `anon_28_7_newyork_real_double_05_26_12_05_03-nojoin0` 路网场景下的训练日志进行深度诊断与收敛性分析。通过 CityFlow 仿真平台采集的 80 轮训练数据，结合数理统计与控制理论方法，对模型的学习曲线、稳态性能及抗扰动能力进行了定量评估。结果表明，模型在第 66 轮进入稳态收敛，最终平均旅行时间（ATT）从初始...
  ```

---
### [TC002] Category: Academic Query (Basic)
- **User Prompt**: *"什么是 MPLight 的 pressure (车道压力) 定义？它和 MaxPressure 有什么区别？"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  好的，作为一名顶尖的交通信号控制研究员和学术主编，我将基于您提供的本地学术文献库上下文以及最新的 arXiv 前沿研究，为您详细解答 MPLight 的 pressure 定义及其与 MaxPressure 的区别。

### MPLight 的 Pressure (车道压力) 定义

MPLight 中的“压力”（Pressure）概念直接源于交通控制理论中的 **MaxPressure 控制理论**。根据 Chen 等人（2020）在论文中的定义，MPLight 将压力作为其强化学习（RL）...
  ```

---
### [TC003] Category: Academic Query (LaTeX & PyTorch)
- **User Prompt**: *"请详细分析 CoLight 协同控制的数学原理，并在回答中给出状态和动作定义的 LaTeX 公式以及它的 PyTorch 核心计算代码骨架。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  好的，作为一名顶尖的交通信号控制研究员和学术主编，我将基于您提供的本地学术文献库上下文（Wei et al. 2019）以及最新的 arXiv 联网检索结果，为您深入剖析 CoLight 协同控制的数学原理，并提供其核心的 PyTorch 代码骨架。

### CoLight 协同控制数学原理深度分析

CoLight (Wei et al., 2019) 的核心创新在于利用**图注意力网络 (Graph Attention Network, GAT)** 来建模交通信号灯之间的动态协同关系。与...
  ```

---
### [TC004] Category: Multi-Seed Diagnostic
- **User Prompt**: *"帮我整体分析一下 benchmark_amp 杭州 5816 的种子数据，生成对比曲线并做收敛性与稳定性分析。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  # 自动化仿真评估报告：杭州 5816 路网多种子收敛性与稳定性分析

## 摘要

本报告基于 CityFlow 仿真平台，对 `benchmark_amp` 场景下杭州 `anon_4_4_hangzhou_real_5816` 路网的三种子运行数据进行全面的收敛性与稳定性分析。通过多种子联合仿真，评估了模型在平均旅行时间（ATT）、车辆吞吐量（Throughput）等核心指标上的表现。结果显示，模型在 80 轮训练后已进入稳态收敛，平均旅行时间均值达 768.67 秒，车辆吞吐量均值达 4...
  ```

---
