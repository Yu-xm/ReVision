#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
from pathlib import Path
import warnings

import numpy as np
from tqdm import tqdm

# 允许打印 warning（你也可以改成 "default" 或 "ignore"）
warnings.filterwarnings("always", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Math Helper Functions (Robust & Float64)
# -----------------------------------------------------------------------------

def l2_normalize(x, eps=1e-10):
    """
    Overflow-safe L2 normalize (row-wise), robust to:
    - very large magnitudes (norm overflow)
    - very small vectors (near-zero norm)
    - NaN/Inf (defensive)
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale each row by max abs value to prevent overflow in norm computation
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    scale = np.where(scale > 0.0, scale, 1.0)

    x_scaled = x / scale
    norm_scaled = np.linalg.norm(x_scaled, axis=1, keepdims=True)

    # Restore true norm
    norm = scale * norm_scaled

    # Defensive sanitize
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.where(norm < eps, 1.0, norm)

    return x / norm


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def ensure_finite(name, arr, pkl_path=None, chunk_start=None):
    """Hard-stop if NaN/Inf appears."""
    if np.isfinite(arr).all():
        return True
    where = []
    if pkl_path is not None:
        where.append(f"file={Path(pkl_path).name}")
    if chunk_start is not None:
        where.append(f"chunk={chunk_start}")
    where = (" " + " ".join(where)) if where else ""
    logging.error(f"[NON-FINITE] {name}{where} contains NaN/Inf -> abort.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Compute embeddings: Trace (Old) - Simple Trace Matching.")

    parser.add_argument("--input_dir", type=str,
                        default="/Users/yu/Desktop/modality_gap/exp_embedding/embedding_bunny/text")
    parser.add_argument("--img_input_dir", type=str,
                        default="/Users/yu/Desktop/modality_gap/exp_embedding/embedding_bunny/image")
    parser.add_argument("--output_dir", type=str,
                        default="/Users/yu/Desktop/modality_gap/exp_embedding/embedding_bunny/text_trace")

    parser.add_argument("--mean_file_path", type=str,
                        default="/Users/yu/Desktop/modality_gap/exp_embedding/embedding_bunny/text_trace/mean_vector.pkl")

    parser.add_argument("--chunk_size", type=int, default=10000)

    # Note: cov_ridge and no_blas_stats are not strictly needed for Trace method 
    # but kept to maintain argument compatibility.
    parser.add_argument("--cov_ridge", type=float, default=1e-6)
    parser.add_argument("--no_blas_stats", type=int, default=1,
                        help="Not used in Trace method, kept for compatibility.")

    # NEW: strict finite checks (stop immediately if any NaN/Inf appears internally)
    parser.add_argument("--strict_finite", type=int, default=1)

    # NEW: save key stats for downstream checking (mean alignment etc.)
    parser.add_argument("--save_stats", type=int, default=1)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    img_input_dir = Path(args.img_input_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = base_output_dir / "process_trace.log"
    setup_logging(log_path)

    text_files = sorted(list(input_dir.glob("*.pkl")))
    img_files = sorted(list(img_input_dir.glob("*.pkl")))

    if not text_files or not img_files:
        logging.error("未找到输入文件。")
        return

    strict = bool(args.strict_finite)

    logging.info(f"模式: Trace (Old) | Text: {len(text_files)} | Image: {len(img_files)}")
    logging.info(f"Options: strict_finite={strict}")

    # =========================================================================
    # Pass 1: 全局均值 (unit-normalized then mean)
    # =========================================================================
    logging.info(">>> Pass 1/4: 计算全局均值...")

    sum_txt = None
    count_txt = 0
    dim = 0

    for pkl_path in tqdm(text_files, desc="Pass 1 (Text Mean)"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if not data:
            continue

        embeds = [np.asarray(x["embed"]) for x in data if x.get("embed") is not None]
        if not embeds:
            continue

        full_mat = np.vstack(embeds).astype(np.float64)

        if dim == 0:
            dim = full_mat.shape[1]
            sum_txt = np.zeros(dim, dtype=np.float64)

        for i in range(0, len(full_mat), args.chunk_size):
            chunk = full_mat[i: i + args.chunk_size]
            chunk = l2_normalize(chunk)
            sum_txt += np.sum(chunk, axis=0)
            count_txt += chunk.shape[0]

    sum_img = None
    count_img = 0

    for pkl_path in tqdm(img_files, desc="Pass 1 (Image Mean)"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if not data:
            continue

        embeds = [np.asarray(x["embed"]) for x in data if x.get("embed") is not None]
        if not embeds:
            continue

        full_mat = np.vstack(embeds).astype(np.float64)

        if sum_img is None:
            sum_img = np.zeros(dim, dtype=np.float64)

        for i in range(0, len(full_mat), args.chunk_size):
            chunk = full_mat[i: i + args.chunk_size]
            chunk = l2_normalize(chunk)
            sum_img += np.sum(chunk, axis=0)
            count_img += chunk.shape[0]

    if count_txt == 0 or count_img == 0:
        logging.error("无数据。")
        return

    global_mu_txt = (sum_txt / count_txt).astype(np.float32)
    global_mu_img = (sum_img / count_img).astype(np.float32)

    # 保存 text mean（兼容你原本 check mean 的脚本）
    try:
        Path(args.mean_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.mean_file_path, "wb") as f:
            pickle.dump(global_mu_txt, f)
    except Exception:
        pass

    # =========================================================================
    # Pass 2: 计算全局 Trace (Variance)
    # =========================================================================
    logging.info(">>> Pass 2/4: 计算全局 Trace (Simple Trace Matching)...")

    mu_t_64 = global_mu_txt.astype(np.float64)
    mu_i_64 = global_mu_img.astype(np.float64)

    # Trace(Old) Logic:
    # 1. 计算 Text 的 trace (sum of squared distances from mean)
    # 2. 计算 Image 的 trace
    # 3. Scale = sqrt(Trace_img / Trace_txt)

    trace_sum_txt = 0.0
    n_txt = 0

    for pkl_path in tqdm(text_files, desc="Pass 2 (Text Trace)"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        embeds = [np.asarray(x["embed"]) for x in data if x.get("embed") is not None]
        if not embeds:
            continue

        X_full = np.vstack(embeds).astype(np.float64)

        for start in range(0, len(X_full), args.chunk_size):
            X = X_full[start: start + args.chunk_size]
            X = l2_normalize(X)

            # Center X
            X_centered = X - mu_t_64
            
            if strict:
                if not ensure_finite("X_centered", X_centered, pkl_path, start): return

            # Sum of squared euclidean norms is equivalent to sum of all elements squared
            trace_sum_txt += np.sum(X_centered ** 2)
            n_txt += X.shape[0]

    trace_sum_img = 0.0
    n_img = 0

    for pkl_path in tqdm(img_files, desc="Pass 2 (Image Trace)"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        embeds = [np.asarray(x["embed"]) for x in data if x.get("embed") is not None]
        if not embeds:
            continue

        X_full = np.vstack(embeds).astype(np.float64)

        for start in range(0, len(X_full), args.chunk_size):
            X = X_full[start: start + args.chunk_size]
            X = l2_normalize(X)

            # Center Image
            X_centered = X - mu_i_64

            if strict:
                if not ensure_finite("X_centered_img", X_centered, pkl_path, start): return

            trace_sum_img += np.sum(X_centered ** 2)
            n_img += X.shape[0]

    logging.info("正在计算 Scale...")

    if n_txt <= 0 or n_img <= 0:
        logging.error("统计量不足，无法计算 Trace Scale。")
        return

    # Average Trace (Mean Squared Distance)
    avg_trace_txt = trace_sum_txt / n_txt
    avg_trace_img = trace_sum_img / n_img

    scale = np.sqrt(avg_trace_img / (avg_trace_txt + 1e-10))

    # -------------------------
    # NEW: finite-check + magnitude log
    # -------------------------
    logging.info(f"[CHECK] avg_trace_txt={avg_trace_txt:.3e}, avg_trace_img={avg_trace_img:.3e}")
    logging.info(f"[CHECK] scale finite? {np.isfinite(scale)} value={scale}")

    if not np.isfinite(scale):
        logging.error("Scale contain NaN/Inf -> 输出不可用，停止。")
        return

    trace_transform_params = {
        "mu_t": mu_t_64,
        "mu_i": mu_i_64,
        "scale": scale,
    }

    # =========================================================================
    # Pass 3: 计算 mu_hat (Mean Correction)
    # Trace(Old) 逻辑通常比较简单，但为了保持 Robust，这里计算变换后的均值
    # 以便在 Pass 4 进行最终校准 (l2 norm 会引起均值漂移)
    # =========================================================================
    logging.info(">>> Pass 3/4: 计算 mu_hat...")

    sum_txt_hat = np.zeros(dim, dtype=np.float64)
    total_hat_count = 0
    params = trace_transform_params

    for pkl_path in tqdm(text_files, desc="Pass 3 (Calc Mu_Hat)"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        embeds = [np.asarray(x["embed"]) for x in data if x.get("embed") is not None]
        if not embeds:
            continue

        X_full = np.vstack(embeds).astype(np.float64)

        for start in range(0, len(X_full), args.chunk_size):
            X = X_full[start: start + args.chunk_size]
            X = l2_normalize(X)

            # Apply Trace Logic: (X - mu_x) * scale + mu_y
            X_centered = X - params["mu_t"]
            X_trans = X_centered * params["scale"] + params["mu_i"]

            X_hat = l2_normalize(X_trans)

            if strict:
                if not ensure_finite("X_hat", X_hat, pkl_path, start): return

            sum_txt_hat += np.sum(X_hat, axis=0)
            total_hat_count += X_hat.shape[0]

    if total_hat_count == 0:
        logging.error("无法计算 mu_hat")
        return

    global_mu_hat_txt = (sum_txt_hat / total_hat_count).astype(np.float64)

    # =========================================================================
    # Pass 4: 应用并保存
    # =========================================================================
    logging.info(">>> Pass 4/4: 应用并保存...")

    out_trace_dir = base_output_dir / "trace"
    out_trace_dir.mkdir(exist_ok=True)

    mu_hat = global_mu_hat_txt
    mu_img = params["mu_i"]

    for pkl_path in tqdm(text_files, desc="Pass 4 (Apply & Save)"):
        file_stem = pkl_path.stem
        file_suffix = pkl_path.suffix

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if not data:
            continue

        valid_items = []
        embeds_list = []
        for item in data:
            emb = item.get("embed")
            if emb is not None:
                valid_items.append(item)
                embeds_list.append(np.asarray(emb))

        if not embeds_list:
            continue

        X_full = np.vstack(embeds_list).astype(np.float64)
        X_final_all = []

        for start in range(0, len(X_full), args.chunk_size):
            X = X_full[start: start + args.chunk_size]
            X = l2_normalize(X)

            # Apply Trace Logic: (X - mu_x) * scale + mu_y
            X_centered = X - params["mu_t"]
            X_trans = X_centered * params["scale"] + params["mu_i"]

            X_hat = l2_normalize(X_trans)

            # Mean Correction (Bias Fix) - 保持脚本一致性
            # 将变换后的数据重新对齐到 mu_img
            X_final_shifted = X_hat - mu_hat + mu_img
            X_final = l2_normalize(X_final_shifted)

            if strict:
                if not ensure_finite("X_final", X_final, pkl_path, start): return

            X_final_all.append(X_final.astype(np.float32))

        X_final_merged = np.vstack(X_final_all)

        res_list = []
        for idx, item in enumerate(valid_items):
            new_item = item.copy()
            new_item["embed"] = X_final_merged[idx]
            res_list.append(new_item)

        out_name = f"{file_stem}_trace{file_suffix}"
        with open(out_trace_dir / out_name, "wb") as f:
            pickle.dump(res_list, f)

    # =========================================================================
    # NEW: Save key stats for downstream validation
    # =========================================================================
    if args.save_stats:
        stats_path = base_output_dir / "trace_stats.pkl"
        stats = {
            "mu_txt": global_mu_txt,                  # float32
            "mu_img": global_mu_img,                  # float32
            "mu_hat": global_mu_hat_txt.astype(np.float32),   # float32
            "scale": float(scale),
            "trace_txt": float(avg_trace_txt),
            "trace_img": float(avg_trace_img),
            "strict_finite": bool(strict),
        }
        try:
            with open(stats_path, "wb") as f:
                pickle.dump(stats, f)
            logging.info(f"已保存关键统计量: {stats_path}")
        except Exception as e:
            logging.warning(f"保存 trace_stats.pkl 失败: {e}")

    logging.info("全部处理完成。")


if __name__ == "__main__":
    main()