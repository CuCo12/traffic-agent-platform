# Traffic Agent Evaluation Harness Run Report

- **Run Timestamp**: 2026-06-16 23:07:05
- **Success Rate**: 100.0% (4/4 Passed)
- **Average Execution Latency**: 26.34s

## Summary Table

| Test Case | Category | Expected Intent | Actual Intent | Image Gen? | Latency | Status | Notes / Failures |
|---|---|---|---|---|---|---|---|
| TC001 | Simulation Diagnostic | `DIAGNOSTIC` | `DIAGNOSTIC` | True | 35.8s | ✅ | N/A |
| TC002 | Academic Query (Basic) | `QUERY_ONLY` | `QUERY_ONLY` | False | 15.3s | ✅ | N/A |
| TC003 | Academic Query (LaTeX & PyTorch) | `QUERY_ONLY` | `QUERY_ONLY` | False | 26.8s | ✅ | N/A |
| TC004 | Multi-Seed Diagnostic | `DIAGNOSTIC` | `DIAGNOSTIC` | True | 27.5s | ✅ | N/A |

## Detailed Test Case Output Previews

### [TC001] Category: Simulation Diagnostic
- **User Prompt**: *"请帮我诊断最新的强化学习控制日志并画出收敛图。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  # 自动化仿真评估报告：基于深度强化学习的交通信号控制收敛性诊断

## 摘要

本报告基于 CityFlow 仿真平台，对最新训练的深度强化学习交通信号控制模型（模型路径：`anon_28_7_newyork_real_double_05_26_12_05_03-nojoin0`）进行了全面的收敛性与性能诊断。通过分析 80 轮训练日志，模型在平均旅行时间（ATT）上实现了 **54.0%** 的显著降低（从 1964.41 秒降至 903.07 秒），平均行车速度提升 **133.2%**，...
  ```

---
### [TC002] Category: Academic Query (Basic)
- **User Prompt**: *"什么是 MPLight 的 pressure (车道压力) 定义？它和 MaxPressure 有什么区别？"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  好的，作为一名交通信号控制研究员，我将为您详细解答 MPLight 中的“压力”（Pressure）定义，并深入剖析其与经典 MaxPressure 控制方法的区别。

### MPLight 中的压力 (Pressure) 定义

MPLight 中的“压力”概念直接源于交通工程领域的 **MaxPressure 控制理论**。在 MPLight 中，压力被用作强化学习（RL）智能体的**状态（State）**和**奖励（Reward）**的核心组成部分。

**1. 车道/相位压力 (Lan...
  ```

---
### [TC003] Category: Academic Query (LaTeX & PyTorch)
- **User Prompt**: *"请详细分析 CoLight 协同控制的数学原理，并在回答中给出状态和动作定义的 LaTeX 公式以及它的 PyTorch 核心计算代码骨架。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  好的，作为一名顶尖的交通信号控制研究员和学术主编，我将为您详细分析 CoLight 协同控制的数学原理，并提供其核心的数学定义与代码实现。

### CoLight: 基于图注意力网络的网络级协同信号控制

CoLight (Wei et al., 2019) 的核心创新在于，它利用图注意力网络（Graph Attention Network, GAT）来建模交通信号灯之间的动态协同关系。与传统的将相邻路口信息简单拼接的方法不同，CoLight 能够学习到不同邻居路口对目标路口影响的**时空动态...
  ```

---
### [TC004] Category: Multi-Seed Diagnostic
- **User Prompt**: *"帮我整体分析一下 benchmark_amp 杭州 5816 的种子数据，生成对比曲线并做收敛性与稳定性分析。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  # 自动化仿真评估报告：杭州 5816 路网多种子收敛性与稳定性分析

## 摘要

本报告基于 CityFlow 仿真平台，对 `benchmark_amp` 场景下杭州 `anon_4_4_hangzhou_real_5816` 路网的三种子运行数据进行全面的收敛性与稳定性评估。通过多种子联合分析，报告揭示了模型在平均旅行时间（ATT）、车辆吞吐量、平均延时、排队长度及停车次数等核心指标上的表现。数理诊断表明，模型在训练第 39 轮后进入稳态，稳态变异系数（CV）为 7.31%，体现了良好的...
  ```

---
