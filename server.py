"""
LTX-2 Video Generation API Server
FastAPI wrapper around LTX-2 distilled pipeline.
Internal use only - not exposed to the internet.
"""

import os
import sys
import uuid
import time
import asyncio
import traceback
import tempfile
from pathlib import Path
from typing import Optional

# Add LTX-2 packages to path (workspace packages)
sys.path.insert(0, "/app/packages/ltx-pipelines/src")
sys.path.insert(0, "/app/packages/ltx-core/src")

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

app = FastAPI(title="LTX-2 Video Generation", version="2.0.0")

MODEL_DIR = Path("/models")
OUTPUT_DIR = Path("/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global pipeline (loaded once)
pipeline = None
pipeline_loading = False
pipeline_error = None


class GenerateRequest(BaseModel):
    prompt: str
    image_path: Optional[str] = None  # Path to input image for img2vid
    image_frame_idx: int = 0  # Which frame to condition on (0=first)
    image_strength: float = 1.0  # Conditioning strength
    width: int = 768  # Will be 2x upsampled (stage 2)
    height: int = 512  # Will be 2x upsampled (stage 2)
    num_frames: int = 97  # ~4 seconds at 24fps
    frame_rate: float = 24.0
    seed: Optional[int] = None
    enhance_prompt: bool = False


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
        "pipeline_loading": pipeline_loading,
        "pipeline_error": pipeline_error,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "models_dir": str(MODEL_DIR),
        "models_present": {
            "checkpoint": (MODEL_DIR / "ltx-2-19b-distilled-fp8.safetensors").exists(),
            "spatial_upsampler": (MODEL_DIR / "ltx-2-spatial-upscaler-x2-1.0.safetensors").exists(),
            "distilled_lora": (MODEL_DIR / "ltx-2-19b-distilled-lora-384.safetensors").exists(),
            "gemma": (MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized" / "config.json").exists(),
        },
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


UPLOAD_DIR = Path("/outputs/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/v1/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for img2vid conditioning. Returns the internal path."""
    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix if file.filename else ".png"
    dest = UPLOAD_DIR / f"{file_id}{ext}"
    content = await file.read()
    dest.write_bytes(content)
    return {"path": str(dest), "size": len(content)}


@app.get("/v1/frame/{task_id}")
async def extract_frame(task_id: str, index: int = -1):
    """Extract a frame from a completed video. index=-1 means last frame."""
    if task_id not in tasks:
        raise HTTPException(404, f"Task {task_id} not found")
    task = tasks[task_id]
    if task.status != "completed" or not task.output_path:
        raise HTTPException(400, f"Task not completed: {task.status}")

    import subprocess
    video_path = task.output_path
    frame_path = str(OUTPUT_DIR / f"{task_id}_frame_{index}.png")

    if index == -1:
        # Extract last frame using ffmpeg
        cmd = [
            "ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
            "-frames:v", "1", "-q:v", "1", frame_path
        ]
    else:
        # Extract specific frame
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
            "-q:v", "1", frame_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(500, f"Frame extraction failed: {result.stderr}")

    return FileResponse(frame_path, media_type="image/png")


@app.get("/v1/frame/{task_id}/upload")
async def extract_frame_and_save(task_id: str, index: int = -1):
    """Extract a frame and save it as an upload (returns internal path for img2vid)."""
    if task_id not in tasks:
        raise HTTPException(404, f"Task {task_id} not found")
    task = tasks[task_id]
    if task.status != "completed" or not task.output_path:
        raise HTTPException(400, f"Task not completed: {task.status}")

    import subprocess
    video_path = task.output_path
    frame_id = str(uuid.uuid4())[:8]
    frame_path = str(UPLOAD_DIR / f"{frame_id}.png")

    if index == -1:
        cmd = [
            "ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
            "-frames:v", "1", "-q:v", "1", frame_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
            "-q:v", "1", frame_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(500, f"Frame extraction failed: {result.stderr}")

    return {"path": frame_path, "task_id": task_id, "frame_index": index}


async def run_generation(task_id: str, req: GenerateRequest):
    """Background generation task."""
    global pipeline, pipeline_loading, pipeline_error

    tasks[task_id].status = "running"
    start_time = time.time()

    try:
        # Lazy-load pipeline
        if pipeline is None and not pipeline_loading:
            pipeline_loading = True
            pipeline_error = None
            print("Loading LTX-2 distilled pipeline...")
            print(f"  Checkpoint: {MODEL_DIR / 'ltx-2-19b-distilled-fp8.safetensors'}")
            print(f"  Gemma: {MODEL_DIR / 'gemma-3-12b-it-qat-q4_0-unquantized'}")
            print(f"  Upsampler: {MODEL_DIR / 'ltx-2-spatial-upscaler-x2-1.0.safetensors'}")

            try:
                import torch
                from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
                from ltx_pipelines.distilled import DistilledPipeline

                lora_path = str(MODEL_DIR / "ltx-2-19b-distilled-lora-384.safetensors")
                loras = [LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]

                # Detect FP8 support (requires Ada Lovelace / sm_89+ or Hopper)
                import torch
                gpu_arch = torch.cuda.get_device_capability()
                use_fp8 = gpu_arch[0] >= 9 or (gpu_arch[0] == 8 and gpu_arch[1] >= 9)
                print(f"  GPU arch: sm_{gpu_arch[0]}{gpu_arch[1]}, FP8: {'yes' if use_fp8 else 'no (using bf16)'}")

                # Pick checkpoint based on FP8 support
                # FP8: use fp8 checkpoint with native fp8 compute (~19GB)
                # No FP8: use fp8 checkpoint but disable fp8 compute (loads as bf16, ~38GB)
                #   -> may OOM on 24GB cards, but sequential offloading helps
                checkpoint = str(MODEL_DIR / "ltx-2-19b-distilled-fp8.safetensors")
                print(f"  Checkpoint: {checkpoint}")
                print(f"  fp8transformer={use_fp8}")

                pipeline = DistilledPipeline(
                    checkpoint_path=checkpoint,
                    gemma_root=str(MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"),
                    spatial_upsampler_path=str(MODEL_DIR / "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
                    loras=loras,
                    fp8transformer=use_fp8,
                )
                print("Pipeline loaded successfully!")
            except Exception as e:
                pipeline_error = str(e)
                print(f"Pipeline load failed: {e}")
                traceback.print_exc()
                raise
            finally:
                pipeline_loading = False

        # Wait if another request is loading the pipeline
        while pipeline_loading:
            await asyncio.sleep(1)

        if pipeline is None:
            raise RuntimeError(f"Failed to load pipeline: {pipeline_error}")

        # Build images list for conditioning (image-to-video support)
        images = []
        if req.image_path and Path(req.image_path).exists():
            images.append((req.image_path, req.image_frame_idx, req.image_strength))

        # Set seed
        seed = req.seed if req.seed is not None else int(time.time()) % 100000

        output_path = str(OUTPUT_DIR / f"{task_id}.mp4")

        print(f"Generating video: {req.prompt[:80]}...")
        print(f"  Size: {req.width}x{req.height}, frames: {req.num_frames}, seed: {seed}")

        # Import needed utilities
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

        tiling_config = TilingConfig.default()

        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()

        def do_generate():
            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=req.prompt,
                    seed=seed,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    frame_rate=req.frame_rate,
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=req.enhance_prompt,
                )

                video_chunks_number = get_video_chunks_number(req.num_frames, tiling_config)
                encode_video(
                    video=video,
                    fps=req.frame_rate,
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )

        await loop.run_in_executor(None, do_generate)

        elapsed = time.time() - start_time
        tasks[task_id].status = "completed"
        tasks[task_id].output_path = output_path
        tasks[task_id].duration_seconds = round(elapsed, 1)
        print(f"Task {task_id} completed in {elapsed:.1f}s → {output_path}")

    except Exception as e:
        elapsed = time.time() - start_time
        # Capture full traceback as string for API response
        tb = traceback.format_exc()
        tasks[task_id].status = "failed"
        tasks[task_id].error = f"{str(e)}\n\nTraceback:\n{tb}"
        tasks[task_id].duration_seconds = round(elapsed, 1)
        print(f"Task {task_id} failed after {elapsed:.1f}s: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    # Use GPU 1 (second 3090) by default
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    uvicorn.run(app, host="0.0.0.0", port=28090)
