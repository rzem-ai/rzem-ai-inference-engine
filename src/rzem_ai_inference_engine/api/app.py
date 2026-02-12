"""FastAPI application factory and WebSocket endpoint."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

from rzem_ai_inference_engine.api.routes import router
from rzem_ai_inference_engine.api.state import JobStateStore
from rzem_ai_inference_engine.api.ws import ConnectionManager
from rzem_ai_inference_engine.types import PreviewConfig


def create_app(
    device: str = "auto",
    vram_limit_gb: float | None = None,
    output_dir: str = "./output",
    preview_config: PreviewConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    announce: bool = True,
) -> FastAPI:
    """Create the FastAPI application with engine lifecycle management."""

    _output_dir = Path(output_dir).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Import engine lazily (pulls in torch)
        from rzem_ai_inference_engine.engine import InferenceEngine

        logger.info(f"Starting inference engine (device={device})")
        engine = InferenceEngine(device=device, vram_limit_gb=vram_limit_gb, preview_config=preview_config)

        ws_manager = ConnectionManager()
        loop = asyncio.get_running_loop()
        store = JobStateStore(engine, ws_manager, _output_dir, loop)

        # Attach to app state for route handlers
        app.state.engine = engine
        app.state.store = store
        app.state.ws_manager = ws_manager

        # Announce on local network via mDNS
        announcer = None
        if announce:
            from rzem_ai_inference_engine.api.announce import ServiceAnnouncer

            announcer = ServiceAnnouncer(
                host=host,
                port=port,
                properties={
                    "version": "0.1.0",
                    "device": device,
                    "api": "rest",
                    "ws": "/ws",
                },
            )
            await announcer.register()

        yield

        if announcer is not None:
            await announcer.unregister()

        logger.info("Shutting down inference engine")
        engine.shutdown()

    app = FastAPI(
        title="RZEM AI Inference Engine API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        manager: ConnectionManager = app.state.ws_manager
        await manager.connect(websocket)
        try:
            # Keep the connection alive; we only broadcast, not receive.
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    return app
