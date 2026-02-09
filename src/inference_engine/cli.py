"""Click-based CLI for the inference engine."""

from __future__ import annotations

import sys
import threading
import time

import click
from loguru import logger

from inference_engine.types import EventType, TransformerType


def parse_lora(value: str) -> tuple[str, float]:
    """Parse a LoRA spec like 'path/to/lora.safetensors:0.8'."""
    if ":" in value:
        parts = value.rsplit(":", 1)
        try:
            strength = float(parts[1])
            return parts[0], strength
        except ValueError:
            pass
    return value, 1.0


@click.group()
def main():
    """Inference engine for text-to-image generation."""
    pass


@main.command()
@click.option("--prompt", required=True, help="Text prompt for image generation")
@click.option("--steps", default=20, help="Number of denoising steps")
@click.option("--cfg-scale", default=1.0, help="Classifier-free guidance scale")
@click.option("--width", default=1024, help="Output image width")
@click.option("--height", default=1024, help="Output image height")
@click.option("--seed", default=-1, help="Random seed (-1 for random)")
@click.option("--transformer-model", required=True, help="Transformer model path or HF repo")
@click.option(
    "--transformer-type",
    required=True,
    type=click.Choice([t.value for t in TransformerType]),
    help="Transformer architecture type",
)
@click.option("--transformer-config", default=None, help="Transformer config (JSON string, file path, or None)")
@click.option("--clip-tokenizer", default=None, help="CLIP tokenizer path or HF repo (FLUX.1)")
@click.option("--clip-tokenizer-config", default=None, help="CLIP tokenizer config")
@click.option("--clip-encoder", default=None, help="CLIP encoder path or HF repo (FLUX.1)")
@click.option("--clip-encoder-config", default=None, help="CLIP encoder config")
@click.option("--t5-tokenizer", default=None, help="T5 tokenizer path or HF repo (FLUX.1)")
@click.option("--t5-tokenizer-config", default=None, help="T5 tokenizer config")
@click.option("--t5-encoder", default=None, help="T5 encoder path or HF repo (FLUX.1)")
@click.option("--t5-encoder-config", default=None, help="T5 encoder config")
@click.option("--qwen3-tokenizer", default=None, help="Qwen3 tokenizer path or HF repo")
@click.option("--qwen3-tokenizer-config", default=None, help="Qwen3 tokenizer config")
@click.option("--qwen3-encoder", default=None, help="Qwen3 encoder path or HF repo")
@click.option("--qwen3-encoder-config", default=None, help="Qwen3 encoder config")
@click.option("--vae-model", required=True, help="VAE model path or HF repo")
@click.option("--vae-config", default=None, help="VAE config")
@click.option("--sampler", default="euler", help="Sampler type (euler, heun, lcm)")
@click.option("--scheduler", default="normal", help="Scheduler type (normal, shift)")
@click.option("--lora", multiple=True, help="LoRA file:strength (e.g. model.safetensors:0.8)")
@click.option("--output", "-o", default="output.png", help="Output image path")
@click.option("--device", default="auto", help="Device (auto, cuda, mps, cpu)")
@click.option("--vram-limit", default=None, type=float, help="VRAM limit in GB")
def generate(
    prompt: str,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    seed: int,
    transformer_model: str,
    transformer_type: str,
    transformer_config: str | None,
    clip_tokenizer: str | None,
    clip_tokenizer_config: str | None,
    clip_encoder: str | None,
    clip_encoder_config: str | None,
    t5_tokenizer: str | None,
    t5_tokenizer_config: str | None,
    t5_encoder: str | None,
    t5_encoder_config: str | None,
    qwen3_tokenizer: str | None,
    qwen3_tokenizer_config: str | None,
    qwen3_encoder: str | None,
    qwen3_encoder_config: str | None,
    vae_model: str,
    vae_config: str | None,
    sampler: str,
    scheduler: str,
    lora: tuple[str, ...],
    output: str,
    device: str,
    vram_limit: float | None,
):
    """Generate an image from a text prompt."""
    from inference_engine.engine import InferenceEngine
    from inference_engine.types import (
        CompletedEvent,
        FailedEvent,
        JobParams,
        LoraParams,
        ProgressEvent,
    )

    # Parse LoRAs
    lora_params = []
    for l in lora:
        path, strength = parse_lora(l)
        lora_params.append(LoraParams(model_file=path, strength=strength))

    # Build job params
    params = JobParams(
        prompt=prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        seed=seed,
        transformer_model=transformer_model,
        transformer_type=TransformerType(transformer_type),
        transformer_config=transformer_config,
        clip_tokenizer=clip_tokenizer,
        clip_tokenizer_config=clip_tokenizer_config,
        clip_encoder=clip_encoder,
        clip_encoder_config=clip_encoder_config,
        t5_tokenizer=t5_tokenizer,
        t5_tokenizer_config=t5_tokenizer_config,
        t5_encoder=t5_encoder,
        t5_encoder_config=t5_encoder_config,
        qwen3_tokenizer=qwen3_tokenizer,
        qwen3_tokenizer_config=qwen3_tokenizer_config,
        qwen3_encoder=qwen3_encoder,
        qwen3_encoder_config=qwen3_encoder_config,
        vae_model=vae_model,
        vae_config=vae_config,
        sampler=sampler,
        scheduler=scheduler,
        loras=lora_params,
    )

    # Create engine
    click.echo(f"Initializing engine (device={device})...")
    engine = InferenceEngine(device=device, vram_limit_gb=vram_limit)

    # Completion tracking
    done_event = threading.Event()
    result: dict = {}

    def on_progress(event: ProgressEvent):
        click.echo(f"  Step {event.step}/{event.total_steps}")

    def on_completed(event: CompletedEvent):
        result["image"] = event.image
        result["seed"] = event.seed
        done_event.set()

    def on_failed(event: FailedEvent):
        result["error"] = event.error
        result["traceback"] = event.traceback
        done_event.set()

    engine.on(EventType.JOB_PROGRESS, on_progress)
    engine.on(EventType.JOB_COMPLETED, on_completed)
    engine.on(EventType.JOB_FAILED, on_failed)

    # Submit and wait
    click.echo(f"Submitting job: {transformer_type}, {steps} steps, {width}x{height}")
    job_id = engine.submit(params)
    click.echo(f"Job ID: {job_id}")

    done_event.wait()

    if "error" in result:
        click.echo(f"Generation failed: {result['error']}", err=True)
        if result.get("traceback"):
            click.echo(result["traceback"], err=True)
        engine.shutdown()
        sys.exit(1)

    # Save image
    image = result["image"]
    image.save(output)
    click.echo(f"Image saved to {output} (seed={result['seed']})")

    engine.shutdown()


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
@click.option("--device", default="auto", help="Device (auto, cuda, mps, cpu)")
@click.option("--vram-limit", default=None, type=float, help="VRAM limit in GB")
@click.option("--output-dir", default="./output", help="Directory for generated images")
@click.option("--preview-interval", default=None, type=int, help="Generate preview every N steps (enables previews)")
@click.option("--preview-size", default=256, type=int, help="Max preview dimension in pixels")
def serve(host: str, port: int, device: str, vram_limit: float | None, output_dir: str, preview_interval: int | None, preview_size: int):
    """Start the REST API server."""
    import uvicorn

    from inference_engine.api import create_app
    from inference_engine.types import PreviewConfig

    preview_config = None
    if preview_interval is not None:
        preview_config = PreviewConfig(enabled=True, interval=preview_interval, max_size=preview_size)

    app = create_app(device=device, vram_limit_gb=vram_limit, output_dir=output_dir, preview_config=preview_config)
    uvicorn.run(app, host=host, port=port)
