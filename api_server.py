import os
import sys
import json
import uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure path to traffic_agent is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from agent import analyze_cityflow_log, search_traffic_literature, read_local_paper_content
from tools.arxiv_searcher import search_latest_arxiv_papers
from tools.plotter import plot_metrics_comparison
from multi_agent_graph import app as graph_app
from langchain_core.messages import HumanMessage

app = FastAPI(
    title="Traffic Agentic Gateway API",
    description="HTTP API gateway to connect local CityFlow parsing, metrics comparison plotting, and academic RAG workflows to Dify.",
    version="1.0.0",
    servers=[{"url": "http://host.docker.internal:8000", "description": "Local Traffic Agentic Gateway"}]
)

# Mount static files to serve the generated images (like traffic_comparison.png)
app.mount("/static", StaticFiles(directory=CURRENT_DIR), name="static")

# Models for request body
class AnalyzeLogRequest(BaseModel):
    log_path: str = Field(..., description="The local absolute path of the simulation log folder containing training_history.json")

class PlotComparisonRequest(BaseModel):
    run1_path: str = Field(..., description="The local absolute path of the baseline simulation log folder")
    run2_path: Optional[str] = Field(None, description="The local absolute path of the optimized simulation log folder (optional)")

class SearchLiteratureRequest(BaseModel):
    query: str = Field(..., description="The academic query regarding traffic signal control algorithms or formulas")

class SearchArxivRequest(BaseModel):
    query: str = Field(..., description="The search query for arXiv, e.g. 'reinforcement learning traffic signal control'")

class ReadPaperRequest(BaseModel):
    filename: str = Field(..., description="The filename of the local paper, e.g. 'AlignLight.pdf'")

class StartEvaluationRequest(BaseModel):
    message: str = Field(..., description="The conversational message or question to run through the first phase of evaluation")

class SubmitApprovalRequest(BaseModel):
    thread_id: str = Field(..., description="The unique thread ID of the paused evaluation run")
    baseline_throughput: float = Field(..., description="The validated or modified throughput of the baseline run")
    baseline_att: float = Field(..., description="The validated or modified ATT of the baseline run")
    optimized_throughput: Optional[float] = Field(None, description="The validated or modified throughput of the optimized run")
    optimized_att: Optional[float] = Field(None, description="The validated or modified ATT of the optimized run")
    review_comment: str = Field("Approved", description="The review comment to resume the workflow with")

class RunGraphRequest(BaseModel):
    message: str = Field(..., description="The conversational message or question to run through the entire multi-agent graph workflow")

# Models for response body
class AnalyzeLogResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    result: Dict = Field(..., description="The parsed traffic metrics including ATT, throughput, etc.")

class PlotComparisonResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    image_path: str = Field(..., description="The local absolute path of the generated image")
    image_url: str = Field(..., description="The HTTP URL to access the comparison image")
    message: str = Field(..., description="Status or feedback message")

class SearchLiteratureResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    result: str = Field(..., description="Retrieved paper paragraphs and RAG context")

class SearchArxivResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    result: str = Field(..., description="The search results and summaries from arXiv")

class ReadPaperResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    result: str = Field(..., description="The complete text content of the requested local paper")

class RunAgentGraphResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    report: str = Field(..., description="The final academic analysis or simulation evaluation report in Markdown format")
    image_url: str = Field(..., description="The HTTP URL of the generated metrics comparison image")

class StartEvaluationResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    thread_id: str = Field(..., description="The unique thread ID of the paused evaluation run")
    is_paused: bool = Field(..., description="Whether the workflow has paused at the human review breakpoint")
    baseline_data: Dict = Field(..., description="The parsed simulation metrics of the baseline run")
    optimized_data: Dict = Field(..., description="The parsed simulation metrics of the optimized run")
    image_url: str = Field(..., description="The HTTP URL of the generated metrics comparison image")
    next_node: List[str] = Field(..., description="The name of the next node in the graph, e.g., ['editor']")

class SubmitApprovalResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    message: str = Field(..., description="Feedback message")
    report: str = Field(..., description="The generated final academic report in Markdown format")

# Router endpoints
@app.post("/analyze_log", summary="Extract CityFlow simulation metrics", response_model=AnalyzeLogResponse)
def api_analyze_log(req: AnalyzeLogRequest):
    """
    Parses and extracts key traffic metrics (like ATT, throughput, etc.) from the training history log folder.
    """
    try:
        res = analyze_cityflow_log.invoke({"log_path": req.log_path})
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plot_comparison", summary="Generate metrics comparison curves", response_model=PlotComparisonResponse)
def api_plot_comparison(req: PlotComparisonRequest, request: Request):
    """
    Generates academic-quality comparison charts for vehicle throughput, average delay, queue length, and speeds, returning the HTTP URL of the image.
    """
    try:
        tool_input = {"run1_path": req.run1_path}
        if req.run2_path:
            tool_input["run2_path"] = req.run2_path
            
        img_path = plot_metrics_comparison.invoke(tool_input)
        # Convert absolute path to a relative static URL
        image_name = os.path.basename(img_path)
        base_url = str(request.base_url)
        image_url = f"{base_url}static/{image_name}"
        return {
            "status": "success",
            "image_path": img_path,
            "image_url": image_url,
            "message": "Comparison chart generated successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_literature", summary="Retrieve literature text using local RAG", response_model=SearchLiteratureResponse)
def api_search_literature(req: SearchLiteratureRequest):
    """
    Searches the local PDF library for traffic signal control algorithms and methods, retrieving relevant paper paragraphs.
    """
    try:
        res = search_traffic_literature.invoke({"query": req.query})
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_arxiv", summary="Search arXiv academic library online", response_model=SearchArxivResponse)
def api_search_arxiv(req: SearchArxivRequest):
    """
    Searches the online arXiv repository for the latest academic papers.
    """
    try:
        res = search_latest_arxiv_papers.invoke({"query": req.query})
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read_paper", summary="Read the full content of a local paper", response_model=ReadPaperResponse)
def api_read_paper(req: ReadPaperRequest):
    """
    Reads and returns the complete text content of a specified local paper in the papers directory.
    """
    try:
        res = read_local_paper_content.invoke({"filename": req.filename})
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_agent_graph", summary="Run multi-agent graph evaluation workflow", response_model=RunAgentGraphResponse)
def api_run_agent_graph(req: RunGraphRequest, request: Request):
    """
    Executes the complete multi-agent graph workflow (Analyst -> Reviewer -> Editor) to generate a full simulation evaluation or academic analysis report.
    """
    try:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "retry_count": 0,
            "baseline_data": {},
            "optimized_data": {},
            "comparison_img": "",
            "review_feedback": "",
            "final_report": ""
        }
        
        # Run graph. It will halt before editor due to interrupt_before
        for event in graph_app.stream(initial_state, config):
            pass
            
        state = graph_app.get_state(config)
        
        # Resume the graph for the editor node
        if state.next:
            for event in graph_app.stream(None, config):
                pass
                
        final_state = graph_app.get_state(config)
        report = final_state.values.get("final_report", "No report generated.")
        comparison_img = final_state.values.get("comparison_img", "")
        image_url = ""
        if comparison_img:
            base_url = str(request.base_url)
            image_url = f"{base_url}static/{os.path.basename(comparison_img)}"
            report += f"\n\n![仿真指标对比图]({image_url})"
            
        return {
            "status": "success",
            "report": report,
            "image_url": image_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_evaluation", summary="Start first phase of evaluation (suspended at human review)", response_model=StartEvaluationResponse)
def api_start_evaluation(req: StartEvaluationRequest, request: Request):
    """
    Runs the multi-agent graph from entry point (Analyst -> Reviewer) and suspends execution at the editor breakpoint.
    Returns the thread_id, parsed metrics, and generated comparison chart.
    """
    try:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "retry_count": 0,
            "baseline_data": {},
            "optimized_data": {},
            "comparison_img": "",
            "review_feedback": "",
            "final_report": ""
        }
        
        # Stream the graph. It will run through analyst and reviewer, then suspend before editor
        for event in graph_app.stream(initial_state, config):
            pass
            
        state = graph_app.get_state(config)
        b_data = state.values.get("baseline_data", {})
        o_data = state.values.get("optimized_data", {})
        comparison_img = state.values.get("comparison_img", "")
        
        image_url = ""
        if comparison_img:
            base_url = str(request.base_url)
            image_url = f"{base_url}static/{os.path.basename(comparison_img)}"
            
        is_paused = bool(state.next)
        
        return {
            "status": "success",
            "thread_id": thread_id,
            "is_paused": is_paused,
            "baseline_data": b_data,
            "optimized_data": o_data,
            "image_url": image_url,
            "next_node": list(state.next) if state.next else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit_approval", summary="Submit human approval/modifications to resume evaluation and compile report", response_model=SubmitApprovalResponse)
def api_submit_approval(req: SubmitApprovalRequest, request: Request):
    """
    Takes the thread_id and validated/modified metrics, updates the state in the graph database, and resumes execution to write the final academic report.
    """
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        state = graph_app.get_state(config)
        if not state.next:
            # Graph already completed or invalid state (e.g. query_only)
            report = state.values.get("final_report", "")
            return {
                "status": "success",
                "message": "Workflow was already completed.",
                "report": report
            }
            
        b_data = state.values.get("baseline_data", {}).copy()
        o_data = state.values.get("optimized_data", {}).copy()
        
        # Override with human modified values
        b_data["throughput"] = req.baseline_throughput
        b_data["avg_delay"] = req.baseline_att
        b_data["raw_analysis"] += f"\n\n[Dify 人工审批微调]: 吞吐量 {req.baseline_throughput}，ATT {req.baseline_att}. 批注: {req.review_comment}"
        
        if o_data and req.optimized_throughput is not None and req.optimized_att is not None:
            o_data["throughput"] = req.optimized_throughput
            o_data["avg_delay"] = req.optimized_att
            o_data["raw_analysis"] += f"\n\n[Dify 人工审批微调]: 吞吐量 {req.optimized_throughput}，ATT {req.optimized_att}."
            
        # Update the graph state as the reviewer node
        graph_app.update_state(
            config,
            {
                "baseline_data": b_data,
                "optimized_data": o_data if o_data else {},
                "review_feedback": "PASS"
            },
            as_node="reviewer"
        )
        
        # Resume execution (will run the editor node and complete)
        for event in graph_app.stream(None, config):
            pass
            
        final_state = graph_app.get_state(config)
        report = final_state.values.get("final_report", "No report generated.")
        comparison_img = final_state.values.get("comparison_img", "")
        if comparison_img:
            base_url = str(request.base_url)
            image_url = f"{base_url}static/{os.path.basename(comparison_img)}"
            report += f"\n\n![仿真指标对比图]({image_url})"
        
        return {
            "status": "success",
            "message": "Workflow resumed and completed successfully.",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Export OpenAPI JSON command
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-schema", action="store_true", help="Export OpenAPI schema to openapi.json and exit")
    args = parser.parse_args()
    
    if args.export_schema:
        # Write OpenAPI schema to openapi.json in the current folder
        schema = app.openapi()
        schema_path = os.path.join(CURRENT_DIR, "openapi.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        print(f"OpenAPI schema successfully exported to: {schema_path}")
    else:
        # Start API server on localhost:8000
        uvicorn.run(app, host="0.0.0.0", port=8000)
