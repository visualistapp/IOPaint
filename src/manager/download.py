import glob
import json
import os
from typing import List

from src.schemas.schema import ModelType, ModelInfo
from loguru import logger
from pathlib import Path


def cli_download_model(model: str):
    from src.models import models
    from src.models.utils import handle_from_pretrained_exceptions

    if model in models and models[model].is_erase_model:
        logger.info(f"Downloading {model}...")
        models[model].download()
        logger.info("Done.")
    else:
        logger.info(f"Downloading model from Huggingface: {model}")
        from diffusers import DiffusionPipeline

        downloaded_path = handle_from_pretrained_exceptions(
            DiffusionPipeline.download, pretrained_model_name=model, variant="fp16"
        )
        logger.info(f"Done. Downloaded to {downloaded_path}")


def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


def scan_diffusers_models() -> List[ModelInfo]:
    from huggingface_hub.constants import HF_HUB_CACHE

    available_models = []
    cache_dir = Path(HF_HUB_CACHE)
    # logger.info(f"Scanning diffusers models in {cache_dir}")
    diffusers_model_names = []
    model_index_files = glob.glob(
        os.path.join(cache_dir, "**/*", "model_index.json"), recursive=True
    )
    for it in model_index_files:
        it = Path(it)
        try:
            with open(it, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue

        _class_name = data["_class_name"]
        name = folder_name_to_show_name(it.parent.parent.parent.name)
        if name in diffusers_model_names:
            continue

        diffusers_model_names.append(name)
        available_models.append(
            ModelInfo(
                name=name,
                path=name,
                model_type=ModelType.DIFFUSERS_OTHER,
            )
        )
    return available_models


def scan_models() -> List[ModelInfo]:
    available_models = []
    available_models.extend(scan_diffusers_models())
    return available_models
