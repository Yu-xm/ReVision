#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_OFFLINE_DEFAULT = ("--online" not in sys.argv)
if _OFFLINE_DEFAULT:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import pickle
import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor
from llm2vec import LLM2Vec

conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    conf.replace("=", ":") if conf else "max_split_size_mb:256"
)

def _is_local_dir(p: str) -> bool:
    try:
        return os.path.isdir(p)
    except Exception:
        return False

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_local_custom_code_present(model_dir: str, offline: bool, required_files: List[str]):
    if not offline:
        return
    if not _is_local_dir(model_dir):
        return

    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"[Offline] config.json not found in local model dir: {model_dir}\n"
            f"Please make sure '{model_dir}' is a valid transformers model directory."
        )

    cfg = _load_json(cfg_path)
    needs_custom = isinstance(cfg, dict) and ("auto_map" in cfg or "custom_code" in str(cfg.get("tags", "")))
    if not needs_custom:
        return

    missing = [fn for fn in required_files if not os.path.isfile(os.path.join(model_dir, fn))]
    if missing:
        raise FileNotFoundError(
            f"[Offline] Missing custom code files in {model_dir}: {missing}"
        )

def infer_dtype_for_device(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps" and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32

def _to_numpy_fp32(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).cpu().numpy()

class FeatureExtractor:
    def __init__(self, llm2clip_path, llm_model_name, device, online: bool = False, llm2vec_code_revision: str = None):
        self.device = device
        self.dtype = infer_dtype_for_device(device)
        self.offline = not online
        self.llm2vec_code_revision = llm2vec_code_revision
        self._img_dim: Optional[int] = None

        print(f"Loading models with dtype: {self.dtype} on {device}...")
        
        self.hub_kwargs = dict(
            trust_remote_code=True,
            local_files_only=self.offline,
        )

        print(f"Loading Vision Encoder from: {llm2clip_path}")
        self.llm2clip = AutoModel.from_pretrained(
            llm2clip_path,
            torch_dtype=self.dtype,
            **self.hub_kwargs,
        ).to(device).eval()

        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                llm2clip_path,
                local_files_only=self.offline,
            )
        except Exception as e:
            print(f"Warning: Could not load processor from {llm2clip_path} ({e})")
            print("Fallback: Using local clip-vit-large-patch14-336")
            self.image_processor = CLIPImageProcessor.from_pretrained(
                "/llm-align/liuchonghan/xiaomin/model/clip-vit-large-patch14-336",
                local_files_only=True,
            )

        print(f"Loading Text Encoder from: {llm_model_name}")
        _ensure_local_custom_code_present(
            llm_model_name,
            offline=self.offline,
            required_files=["modeling_llama_encoder.py"],
        )

        text_kwargs = dict(self.hub_kwargs)
        if (not _is_local_dir(llm_model_name)) and self.llm2vec_code_revision:
            text_kwargs["revision"] = self.llm2vec_code_revision

        cfg = AutoConfig.from_pretrained(llm_model_name, **text_kwargs)

        required_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        if getattr(cfg, "_name_or_path", "") != required_name:
            cfg._name_or_path = required_name

        try:
            cfg._attn_implementation = "sdpa"
        except Exception:
            pass

        self.llm_model = AutoModel.from_pretrained(
            llm_model_name,
            torch_dtype=self.dtype,
            config=cfg,
            **text_kwargs,
        ).to(device).eval()

        tok = AutoTokenizer.from_pretrained(llm_model_name, **text_kwargs)

        self.l2v = LLM2Vec(
            self.llm_model,
            tok,
            pooling_mode="mean",
            max_length=512,
            doc_max_length=512
        )

    def _ensure_img_dim(self):
        if self._img_dim is not None:
            return

        size = getattr(self.image_processor, "size", None)
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                h, w = int(size["height"]), int(size["width"])
            elif "shortest_edge" in size:
                h = w = int(size["shortest_edge"])
            else:
                h = w = 336
        elif isinstance(size, int):
            h = w = int(size)
        else:
            h = w = 336

        dummy = Image.new("RGB", (w, h), (0, 0, 0))
        inputs = self.image_processor(images=[dummy], return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            feats = self.llm2clip.get_image_features(inputs)
        self._img_dim = int(feats.shape[-1])

    @torch.no_grad()
    def encode_texts(self, text_list: List[str]):
        llm_feats = self.l2v.encode(text_list, convert_to_tensor=True)
        llm_feats = llm_feats.to(device=self.device, dtype=self.dtype)

        clip_feats = self.llm2clip.get_text_features(llm_feats)

        den = clip_feats.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clip_feats = clip_feats / den.to(dtype=clip_feats.dtype)

        return _to_numpy_fp32(clip_feats)

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]):
        images = []
        valid_indices = []

        for idx, p in enumerate(image_paths):
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
                valid_indices.append(idx)
            except (OSError, UnidentifiedImageError, FileNotFoundError):
                images.append(None)

        real_images = [img for img in images if img is not None]
        if not real_images:
            self._ensure_img_dim()
            return np.zeros((len(image_paths), self._img_dim), dtype=np.float32)

        inputs = self.image_processor(images=real_images, return_tensors="pt").pixel_values.to(
            self.device, dtype=self.dtype
        )

        img_feats = self.llm2clip.get_image_features(inputs)

        den = img_feats.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        img_feats = img_feats / den.to(dtype=img_feats.dtype)

        self._img_dim = int(img_feats.shape[-1])
        img_feats_np = _to_numpy_fp32(img_feats)

        final_feats = np.zeros((len(image_paths), img_feats_np.shape[1]), dtype=img_feats_np.dtype)
        for real_idx, orig_idx in enumerate(valid_indices):
            final_feats[orig_idx] = img_feats_np[real_idx]

        return final_feats


def batched(seq, bsz):
    for i in range(0, len(seq), bsz):
        yield seq[i: i + bsz]


def save_list_pkl(path: str, items: list):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(items, f)


def process_modality(modality_name, samples, extractor, args, output_dir, save_prefix):
    print(f"\n[Processing {modality_name}] Total samples: {len(samples)}")
    
    finished_ids = set()
    part = 1
    
    out_pattern = str(Path(output_dir) / f"{save_prefix}_*.pkl")
    try:
        existing_files = sorted(glob.glob(out_pattern), key=lambda x: int(Path(x).stem.split('_')[-1]))
    except Exception:
        existing_files = sorted(glob.glob(out_pattern))

    current_buffer = []

    if existing_files and not args.overwrite:
        print(f"Found {len(existing_files)} existing files. Checking for resume...")
        
        last_file_path = existing_files[-1]
        prev_files = existing_files[:-1]

        for p in tqdm(prev_files, desc="Loading completed parts"):
            try:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                    for item in data:
                        finished_ids.add(item["id"])
            except Exception:
                pass
        
        try:
            last_part_num = int(Path(last_file_path).stem.split('_')[-1])
            with open(last_file_path, "rb") as f:
                loaded_items = pickle.load(f)
            
            if len(loaded_items) >= args.records_per_pkl:
                print(f"Last part is full ({len(loaded_items)}). Starting next part.")
                for item in loaded_items:
                    finished_ids.add(item["id"])
                part = last_part_num + 1
            else:
                print(f"Last part is PARTIAL ({len(loaded_items)}/{args.records_per_pkl}). Appending to it.")
                current_buffer = loaded_items
                for item in loaded_items:
                    finished_ids.add(item["id"])
                part = last_part_num
        except Exception:
            part = last_part_num
    else:
        if args.overwrite:
            print("Overwrite mode enabled.")
        else:
            print("No existing files. Starting fresh.")

    print(f"Resuming {modality_name}... Skipped {len(finished_ids)} IDs. Buffer: {len(current_buffer)}. Part {part}")

    for batch in tqdm(batched(samples, args.bsz), total=(len(samples) + args.bsz - 1) // args.bsz):
        batch = [s for s in batch if s["id"] not in finished_ids]
        if not batch:
            continue
        
        ids = [s["id"] for s in batch]

        if modality_name == "IMAGES":
            img_paths = [s["image_path"] for s in batch]
            vecs = extractor.encode_images(img_paths).astype(np.float16)
            
            for sid, p, v in zip(ids, img_paths, vecs):
                if np.all(v == 0): 
                    continue
                current_buffer.append({"id": sid, "image_path": p, "embed": v})
        
        elif modality_name == "TEXT":
            texts = [s["text"] for s in batch]
            vecs = extractor.encode_texts(texts).astype(np.float16)
            
            for sid, t, v in zip(ids, texts, vecs):
                current_buffer.append({"id": sid, "caption": t, "embed": v})
        
        del vecs
        torch.cuda.empty_cache()
        gc.collect()

        if len(current_buffer) >= args.records_per_pkl:
            save_path = Path(output_dir) / f"{save_prefix}_{part}.pkl"
            save_list_pkl(save_path, current_buffer)
            print(f"Saved {modality_name} Part {part}: {len(current_buffer)} records.")
            current_buffer = []
            part += 1

    if current_buffer:
        save_path = Path(output_dir) / f"{save_prefix}_{part}.pkl"
        save_list_pkl(save_path, current_buffer)
        print(f"Saved {modality_name} Final Part {part}: {len(current_buffer)} records.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--output_text_dir", type=str, required=True)
    ap.add_argument("--output_image_dir", type=str, required=True)
    ap.add_argument("--llm2clip_path", type=str, default="/llm-align/liuchonghan/xiaomin/model/llm2clip-openai")
    ap.add_argument("--llm_model_name", type=str, default="/llm-align/liuchonghan/xiaomin/model/llm2clip")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--llm2vec_code_revision", type=str, default="33d5ca9d51c695fa9dcb7e35889e2ee6051cc20a")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bsz", type=int, default=1024)
    ap.add_argument("--records_per_pkl", type=int, default=200000)
    ap.add_argument("--modality", type=str, default="both", choices=["both", "text", "image"])
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    print(f"Reading {args.data_json}...")
    with open(args.data_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    img_samples = []
    txt_samples = []

    if args.modality in ["both", "image"]:
        raw_images = raw_data.get("images", [])
        print(f"Found {len(raw_images)} image entries.")
        for item in raw_images:
            sid = item.get("id")
            path = item.get("image")
            if sid and path:
                full_path = os.path.join(args.image_root, path)
                img_samples.append({"id": sid, "image_path": full_path})

    if args.modality in ["both", "text"]:
        raw_texts = raw_data.get("texts", [])
        print(f"Found {len(raw_texts)} text entries.")
        for item in raw_texts:
            sid = item.get("id")
            txt = item.get("text")
            if sid and txt and len(txt.strip()) > 0:
                txt_samples.append({"id": sid, "text": txt})

    extractor = FeatureExtractor(
        args.llm2clip_path,
        args.llm_model_name,
        torch.device(args.device),
        online=args.online,
        llm2vec_code_revision=args.llm2vec_code_revision,
    )

    if args.modality in ["both", "image"] and img_samples:
        process_modality("IMAGES", img_samples, extractor, args, args.output_image_dir, "image_embeds")
    
    if args.modality in ["both", "text"] and txt_samples:
        process_modality("TEXT", txt_samples, extractor, args, args.output_text_dir, "text_embeds")

    print("\nAll done.")

if __name__ == "__main__":
    main()