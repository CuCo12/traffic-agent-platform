# Traffic Agent Evaluation Harness Run Report

- **Run Timestamp**: 2026-06-16 15:56:16
- **Success Rate**: 100.0% (4/4 Passed)
- **Average Execution Latency**: 25.53s

## Summary Table

| Test Case | Category | Expected Intent | Actual Intent | Image Gen? | Latency | Status | Notes / Failures |
|---|---|---|---|---|---|---|---|
| TC001 | Simulation Diagnostic | `DIAGNOSTIC` | `DIAGNOSTIC` | True | 25.0s | ✅ | N/A |
| TC002 | Academic Query (Basic) | `QUERY_ONLY` | `QUERY_ONLY` | False | 18.7s | ✅ | N/A |
| TC003 | Academic Query (LaTeX & PyTorch) | `QUERY_ONLY` | `QUERY_ONLY` | False | 30.6s | ✅ | N/A |
| TC004 | Multi-Seed Diagnostic | `DIAGNOSTIC` | `DIAGNOSTIC` | True | 27.8s | ✅ | N/A |

## Detailed Test Case Output Previews

### [TC001] Category: Simulation Diagnostic
- **User Prompt**: *"请帮我诊断最新的强化学习控制日志并画出收敛图。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  # 自动化仿真评估报告：基于深度强化学习的交通信号控制收敛性诊断

## 摘要

本报告针对基于深度强化学习（DRL）的交通信号控制模型在 `anon_28_7_newyork_real_double_05_26_12_05_03-nojoin0` 路网场景下的训练日志进行系统性诊断。通过分析 80 轮训练过程中的关键性能指标（平均旅行时间、平均行车速度、平均排队长度、总停车次数、车辆吞吐量），并结合学习曲线的收敛性与稳定性数理诊断，评估模型的控制性能与训练质量。结果表明，模型在第 66 轮后进...
  ```

---
### [TC002] Category: Academic Query (Basic)
- **User Prompt**: *"什么是 MPLight 的 pressure (车道压力) 定义？它和 MaxPressure 有什么区别？"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  好的，作为一名交通信号控制研究员，我将为您详细解答 MPLight 中的“压力 (Pressure)”定义及其与经典 MaxPressure 控制的区别。

### MPLight 中的压力 (Pressure) 定义

MPLight (Chen et al., 2020) 的核心创新在于将交通工程中经典的 **MaxPressure 控制理论** 与 **深度强化学习 (Deep RL)** 相结合。其“压力”的定义直接源于 MaxPressure 理论，但应用方式有所不同。

**1. 交...
  ```

---
### [TC003] Category: Academic Query (LaTeX & PyTorch)
- **User Prompt**: *"请详细分析 CoLight 协同控制的数学原理，并在回答中给出状态和动作定义的 LaTeX 公式以及它的 PyTorch 核心计算代码骨架。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  好的，作为一名顶尖的交通信号控制研究员和学术主编，我将为您详细解析 CoLight 协同控制的数学原理，并提供其核心的数学定义与代码实现。

### CoLight: 基于图注意力网络的协同信号控制原理深度解析

CoLight (Wei et al., 2019) 的核心创新在于，它利用**图注意力网络 (Graph Attention Network, GAT)** 来建模交通信号灯之间的动态协同关系。与传统的将相邻路口信息简单拼接的方法不同，CoLight 能够学习到不同邻居路口对目标路口...
  ```

---
### [TC004] Category: Multi-Seed Diagnostic
- **User Prompt**: *"帮我整体分析一下 benchmark_amp 杭州 5816 的种子数据，生成对比曲线并做收敛性与稳定性分析。"*
- **Execution Status**: ✅ PASSED
- **Output Report Preview**:
  ```markdown
  # 自动化仿真评估报告：杭州 5816 路网多种子联合收敛性与稳定性分析

## 摘要

本报告基于 CityFlow 仿真平台，对 `benchmark_amp` 场景下杭州 `anon_4_4_hangzhou_real_5816` 路网的三种子运行数据进行联合分析。通过多种子聚合统计与单种子学习曲线深度诊断，系统评估了模型在平均旅行时间（ATT）、车辆吞吐量（Throughput）等核心指标上的收敛性能与稳态稳定性。分析表明，模型在 80 轮训练后已进入稳态收敛，平均旅行时间较初始阶段下降...
  ```

---
