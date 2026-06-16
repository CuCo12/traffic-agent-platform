import sys
import os
import time
import uuid
import json
from datetime import datetime

# Ensure traffic_agent directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from multi_agent_graph import app as graph_app
from langchain_core.messages import HumanMessage

# Define the Golden Dataset for Agent Evaluation
GOLDEN_DATASET = [
    {
        "id": "TC001",
        "category": "Simulation Diagnostic",
        "description": "Verify log diagnostic routing and metric extraction on latest simulation history.",
        "input_message": "请帮我诊断最新的强化学习控制日志并画出收敛图。",
        "expected_intent": "DIAGNOSTIC",
        "required_keys": [["平均旅行时间", "ATT"], ["吞吐量", "Throughput"]],
        "check_image": True
    },
    {
        "id": "TC002",
        "category": "Academic Query (Basic)",
        "description": "Check basic academic question routing and prompt-based on-demand rules.",
        "input_message": "什么是 MPLight 的 pressure (车道压力) 定义？它和 MaxPressure 有什么区别？",
        "expected_intent": "QUERY_ONLY",
        "required_keys": [["MPLight"], ["MaxPressure"]],
        "check_image": False
    },
    {
        "id": "TC003",
        "category": "Academic Query (LaTeX & PyTorch)",
        "description": "Verify on-demand LaTeX math equations and PyTorch code skeleton generation.",
        "input_message": "请详细分析 CoLight 协同控制的数学原理，并在回答中给出状态和动作定义的 LaTeX 公式以及它的 PyTorch 核心计算代码骨架。",
        "expected_intent": "QUERY_ONLY",
        "required_keys": [["CoLight"], ["class "], ["torch"], ["$$", "\\[", "\\(", "$"]],
        "check_image": False
    },
    {
        "id": "TC004",
        "category": "Multi-Seed Diagnostic",
        "description": "Verify matching of multiple seeds and average/std metrics analysis.",
        "input_message": "帮我整体分析一下 benchmark_amp 杭州 5816 的种子数据，生成对比曲线并做收敛性与稳定性分析。",
        "expected_intent": "DIAGNOSTIC",
        "required_keys": [["变异系数", "CV"], ["稳态", "收敛"], ["多种子", "种子"]],
        "check_image": True
    }
]

def run_agent_test(test_case):
    print(f"\n==================================================")
    print(f"🚀 Running Test Case [{test_case['id']}] - {test_case['category']}")
    print(f"💬 Input: '{test_case['input_message']}'")
    print(f"==================================================")
    
    config = {"configurable": {"thread_id": f"eval_harness_{uuid.uuid4()}"}}
    initial_state = {
        "messages": [HumanMessage(content=test_case["input_message"])],
        "retry_count": 0,
        "baseline_data": {},
        "optimized_data": {},
        "comparison_img": "",
        "review_feedback": "",
        "final_report": ""
    }
    
    start_time = time.time()
    
    # Phase 1: Run until interrupt
    print("⏳ Running Phase 1 (Data Analyst & Reviewer)...")
    for event in graph_app.stream(initial_state, config):
        pass
        
    state = graph_app.get_state(config)
    
    # Phase 2: Resume (runs the editor node and completes)
    print("⏳ Running Phase 2 (Academic Editor)...")
    if state.next:
        for event in graph_app.stream(None, config):
            pass
            
    end_time = time.time()
    latency = end_time - start_time
    
    final_state = graph_app.get_state(config)
    values = final_state.values
    
    report = values.get("final_report", "")
    img_path = values.get("comparison_img", "")
    baseline_data = values.get("baseline_data", {})
    
    # Determine the intent classified by the Analyst node
    actual_intent = "QUERY_ONLY"
    if baseline_data and baseline_data.get("status") == "computed":
        actual_intent = "DIAGNOSTIC"
        
    print(f"⏱️ Finished in {latency:.2f} seconds.")
    print(f"🎯 Classified Intent: {actual_intent} (Expected: {test_case['expected_intent']})")
    print(f"🖼️ Generated Image: '{os.path.basename(img_path) if img_path else 'None'}'")
    
    # Evaluate assertions
    intent_ok = (actual_intent == test_case["expected_intent"])
    
    image_ok = True
    if test_case["check_image"]:
        image_ok = bool(img_path and os.path.exists(img_path))
    else:
        image_ok = not bool(img_path)
        
    content_ok = True
    missing_keys = []
    for key_list in test_case["required_keys"]:
        any_found = False
        for alt in key_list:
            if alt in report:
                any_found = True
                break
        if not any_found:
            content_ok = False
            missing_keys.append(f"({' or '.join(key_list)})")
            
    # Write the full report to a file for user review
    out_file = os.path.join(CURRENT_DIR, f"eval_output_{test_case['id']}.md")
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📄 Saved full report to: {out_file}")
    except Exception as e:
        print(f"⚠️ Failed to save full report to {out_file}: {e}")
            
    passed = intent_ok and image_ok and content_ok
    status = "✅ PASSED" if passed else "❌ FAILED"
    
    reasons = []
    if not intent_ok:
        reasons.append(f"Intent mismatch (got {actual_intent})")
    if not image_ok:
        reasons.append("Image generation mismatch" if test_case["check_image"] else "Unexpected image generated")
    if not content_ok:
        reasons.append(f"Missing required keywords: {missing_keys}")
        
    print(f"Result: {status}")
    if not passed:
        print(f"⚠️ Failure Reasons: {', '.join(reasons)}")
        
    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "input": test_case["input_message"],
        "latency": latency,
        "actual_intent": actual_intent,
        "expected_intent": test_case["expected_intent"],
        "image_generated": bool(img_path),
        "passed": passed,
        "reasons": reasons,
        "report_preview": report[:250] + "..." if len(report) > 250 else report
    }

def main():
    print("=" * 60)
    print("🚦 Traffic Agentic Platform - Evaluation Harness 🚦")
    print("=" * 60)
    print(f"Starting evaluation run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total test cases to execute: {len(GOLDEN_DATASET)}")
    
    results = []
    for tc in GOLDEN_DATASET:
        res = run_agent_test(tc)
        results.append(res)
        
    # Compile the final summary markdown report
    passed_count = sum(1 for r in results if r["passed"])
    success_rate = (passed_count / len(results)) * 100
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    report_path = os.path.join(CURRENT_DIR, "agent_evaluation_report.md")
    
    # Build Markdown table
    table_rows = []
    for r in results:
        status_emoji = "✅" if r["passed"] else "❌"
        reasons_str = "; ".join(r["reasons"]) if r["reasons"] else "N/A"
        table_rows.append(
            f"| {r['id']} | {r['category']} | `{r['expected_intent']}` | `{r['actual_intent']}` | {r['image_generated']} | {r['latency']:.1f}s | {status_emoji} | {reasons_str} |"
        )
        
    markdown_content = f"""# Traffic Agent Evaluation Harness Run Report

- **Run Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Success Rate**: {success_rate:.1f}% ({passed_count}/{len(results)} Passed)
- **Average Execution Latency**: {avg_latency:.2f}s

## Summary Table

| Test Case | Category | Expected Intent | Actual Intent | Image Gen? | Latency | Status | Notes / Failures |
|---|---|---|---|---|---|---|---|
{"\n".join(table_rows)}

## Detailed Test Case Output Previews

"""
    for r in results:
        markdown_content += f"""### [{r['id']}] Category: {r['category']}
- **User Prompt**: *"{r['input']}"*
- **Execution Status**: {"✅ PASSED" if r['passed'] else "❌ FAILED"}
- **Output Report Preview**:
  ```markdown
  {r['report_preview']}
  ```

---
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print("\n" + "=" * 60)
    print("🏁 Evaluation Run Completed!")
    print(f"Success Rate: {success_rate:.1f}% ({passed_count}/{len(results)} Passed)")
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"Detailed Markdown evaluation report saved to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
