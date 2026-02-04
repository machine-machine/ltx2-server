"""
LTX-2 Video Generation API Server
Simple FastAPI wrapper around LTX-2 pipelines.
Internal use only - not exposed to the internet.
"""

import os
import sys
import uuid
import time
import asyncio
import traceback
from pathlib import Path
from typing import Optional

# Add LTX-2 packages to path (workspace packages)
sys.path.insert(0, "/app/packages/ltx-pipelines/src")
sys.path.insert(0, "/app/packages/ltx-core/src")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="LTX-2 Video Generation", version="1.0.0")

MODEL_DIR = Path("/models")
OUTPUT_DIR = Path("/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global pipeline (loaded once)
pipeline = None
pipeline_loading = False

class GenerateRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    end_image_url: Optional[str] = None
    width: int = 768
    height: int = 512
    num_frames: int = 97  # ~4 seconds at 24fps
    num_steps: int = 8    # Distilled model uses 8 steps
    seed: Optional[int] = None
    guidance_scale: float = 1.0  # CFG=1 for distilled

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    output_path: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

# In-memory task store
tasks: dict[str, TaskStatus] = {}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }


@app.post("/v1/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Submit a video generation job. Returns task_id for polling."""
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = TaskStatus(task_id=task_id, status="pending")
    
    background_tasks.add_task(run_generation, task_id, req)
    
    return {"task_id": task_id, "status": "pending"}


@app.get("/v1/status/{task_id}")
async def status(task_id: str):
    """Poll task status."""
    if task_id not in tasks:
        raise HTTPException(404, f"Task {task_id} not found")
    return tasks[task_id]


@app.get("/v1/download/{task_id}")
async def download(task_id: str):
    """Download completed video."""
    if task_id not in tasks:
        raise HTTPException(404, f"Task {task_id} not found")
    task = tasks[task_id]
    if task.status != "completed" or not task.output_path:
        raise HTTPException(400, f"Task not completed: {task.status}")
    return FileResponse(task.output_path, media_type="video/mp4")


async def run_generation(task_id: str, req: GenerateRequest):
    """Background generation task."""
    global pipeline, pipeline_loading
    
    tasks[task_id].status = "running"
    start_time = time.time()
    
    try:
        # Lazy-load pipeline
        if pipeline is None and not pipeline_loading:
            pipeline_loading = True
            print("Loading LTX-2 distilled pipeline...")
            
            from ltx_pipelines.distilled import DistilledPipeline
            
            pipeline = DistilledPipeline(
                ckpt_path=str(MODEL_DIR / "ltx-2-19b-distilled-fp8.safetensors"),
                text_encoder_path=str(MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"),
            )
            pipeline_loading = False
            print("Pipeline loaded!")
        
        # Wait if another request is loading the pipeline
        while pipeline_loading:
            await asyncio.sleep(1)
        
        if pipeline is None:
            raise RuntimeError("Failed to load pipeline")
        
        # Generate
        output_path = str(OUTPUT_DIR / f"{task_id}.mp4")
        
        kwargs = {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "num_frames": req.num_frames,
            "num_inference_steps": req.num_steps,
            "guidance_scale": req.guidance_scale,
            "output_path": output_path,
        }
        
        if req.seed is not None:
            kwargs["seed"] = req.seed
        
        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pipeline(**kwargs))
        
        elapsed = time.time() - start_time
        tasks[task_id].status = "completed"
        tasks[task_id].output_path = output_path
        tasks[task_id].duration_seconds = round(elapsed, 1)
        print(f"Task {task_id} completed in {elapsed:.1f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        tasks[task_id].status = "failed"
        tasks[task_id].error = str(e)
        tasks[task_id].duration_seconds = round(elapsed, 1)
        print(f"Task {task_id} failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    
    # Use GPU 1 (second 3090) by default
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    uvicorn.run(app, host="0.0.0.0", port=28090)
