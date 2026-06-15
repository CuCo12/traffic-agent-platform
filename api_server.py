import os
import sys
import json
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure path to traffic_agent is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from agent import analyze_cityflow_log, search_traffic_literature
from skills.plotter import plot_metrics_comparison
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

class RunGraphRequest(BaseModel):
    message: str = Field(..., description="The conversational message or question to run through the entire multi-agent graph workflow")

# Router endpoints
@app.post("/analyze_log", summary="Extract CityFlow simulation metrics")
def api_analyze_log(req: AnalyzeLogRequest):
    """
    Parses and extracts key traffic metrics (like ATT, throughput, etc.) from the training history log folder.
    """
    try:
        res = analyze_cityflow_log.invoke({"log_path": req.log_path})
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plot_comparison", summary="Generate metrics comparison curves")
def api_plot_comparison(req: PlotComparisonRequest):
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
        image_url = f"http://localhost:8000/static/{image_name}"
        return {
            "status": "success",
            "image_path": img_path,
            "image_url": image_url,
            "message": "Comparison chart generated successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_literature", summary="Retrieve literature text using local RAG")
def api_search_literature(req: SearchLiteratureRequest):
    """
    Searches the local PDF library for traffic signal control algorithms and methods, retrieving relevant paper paragraphs.
    """
    try:
        res = search_traffic_literature.invoke({"query": req.query})
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_agent_graph", summary="Run multi-agent graph evaluation workflow")
def api_run_agent_graph(req: RunGraphRequest):
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
            image_url = f"http://localhost:8000/static/{os.path.basename(comparison_img)}"
            
        return {
            "status": "success",
            "report": report,
            "image_url": image_url
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
