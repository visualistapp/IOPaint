import asyncio
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import socketio
import torch

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)
except:
    pass

import uvicorn
from PIL import Image
from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from socketio import AsyncServer

from src.utils.helper import (
    load_img,
    decode_base64_to_image,
    pil_to_bytes,
    numpy_to_bytes,
    concat_alpha_channel,
    gen_frontend_mask,
    adjust_mask,
)
from src.model.utils import torch_gc
from src.manager.model_manager import ModelManager
from src.schemas.schema import (
    GenInfoResponse,
    ApiConfig,
    ServerConfigResponse,
    SwitchModelRequest,
    InpaintRequest,
    SDSampler,
    AdjustMaskRequest,
    RemoveBGModel,
    ModelInfo,
    InteractiveSegModel,
    RealESRGANModel,
)

CURRENT_DIR = Path(__file__).parent.absolute().resolve()
WEB_APP_DIR = CURRENT_DIR / "web_app"


def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get("WEBUI_RICH_EXCEPTIONS", None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console

            console = Console()
            rich_available = True
    except Exception:
        pass

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get("detail", ""),
            "body": vars(e).get("body", ""),
            "errors": str(e),
        }
        if not isinstance(
            e, HTTPException
        ):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(
                    show_locals=True,
                    max_frames=2,
                    extra_lines=1,
                    suppress=[anyio, starlette],
                    word_wrap=False,
                    width=min([console.width, 200]),
                )
            else:
                traceback.print_exc()
        return JSONResponse(
            status_code=vars(e).get("status_code", 500), content=jsonable_encoder(err)
        )

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_origins": ["*"],
        "allow_credentials": True,
        "expose_headers": ["X-Seed"],
    }
    app.add_middleware(CORSMiddleware, **cors_options)


global_sio: AsyncServer = None


def diffuser_callback(pipe, step: int, timestep: int, callback_kwargs: Dict = {}):
    # self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict
    # logger.info(f"diffusion callback: step={step}, timestep={timestep}")

    # We use asyncio loos for task processing. Perhaps in the future, we can add a processing queue similar to InvokeAI,
    # but for now let's just start a separate event loop. It shouldn't make a difference for single person use
    asyncio.run(global_sio.emit("diffusion_progress", {"step": step}))
    return {}


class Api:
    def __init__(self, app: FastAPI, config: ApiConfig):
        self.app = app
        self.config = config
        self.router = APIRouter()
        self.queue_lock = threading.Lock()
        self.request_queue = Queue()
        api_middleware(self.app)

        self.model_manager = self._build_model_manager()

        # Start the background worker
        self.worker_thread = Thread(target=self._process_queue)
        self.worker_thread.start()

        # fmt: off
        self.add_api_route("/api/v1/inpaint", self.api_inpaint, methods=["POST"])
        # fmt: on

        global global_sio
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.combined_asgi_app = socketio.ASGIApp(self.sio, self.app)
        self.app.mount("/ws", self.combined_asgi_app)
        global_sio = self.sio

    def _process_queue(self):
        while True:
            request, response_future = self.request_queue.get()
            try:
                response = request()
                response_future.set_result(response)
            except Exception as e:
                response_future.set_exception(e)
            self.request_queue.task_done()

    def _enqueue_request(self, request):
        response_future = asyncio.Future()
        self.request_queue.put((request, response_future))
        return response_future

    async def api_inpaint(self, req: InpaintRequest):
        def process_request():
            image, alpha_channel, infos = decode_base64_to_image(req.image)
            mask, _, _ = decode_base64_to_image(req.mask, gray=True)

            mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            if image.shape[:2] != mask.shape[:2]:
                raise HTTPException(
                    400,
                    detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
                )

            if req.paint_by_example_example_image:
                paint_by_example_image, _, _ = decode_base64_to_image(
                    req.paint_by_example_example_image
                )

            start = time.time()
            rgb_np_img = self.model_manager(image, mask, req)
            logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
            torch_gc()

            rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

            ext = "png"
            res_img_bytes = pil_to_bytes(
                Image.fromarray(rgb_res),
                ext=ext,
                quality=self.config.quality,
                infos=infos,
            )

            asyncio.run(self.sio.emit("diffusion_finish"))

            return Response(
                content=res_img_bytes,
                media_type=f"image/{ext}",
                headers={"X-Seed": str(req.sd_seed)},
            )

        response_future = self._enqueue_request(process_request)
        return await response_future
    
    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def launch(self):
        self.app.include_router(self.router)
        uvicorn.run(
            self.combined_asgi_app,
            host=self.config.host,
            port=self.config.port,
            timeout_keep_alive=999999999,
        )

    def _build_model_manager(self):
        return ModelManager(
            name=self.config.model,
            device=torch.device(self.config.device),
            no_half=self.config.no_half,
            low_mem=self.config.low_mem,
            disable_nsfw=self.config.disable_nsfw_checker,
            sd_cpu_textencoder=self.config.cpu_textencoder,
            local_files_only=self.config.local_files_only,
            cpu_offload=self.config.cpu_offload,
            callback=diffuser_callback,
        )
