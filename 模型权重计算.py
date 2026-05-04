"""
模型组合搜索 — 从 models/ 中穷举最优集成组合

用法: python 模型权重计算.py
"""
import sys
import os
from pathlib import Path
from itertools import combinations
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


MODEL_DIR = Path("models")
TRAIN_FILE = "arknights_filtered.csv"
VAL_FILE = "arknights_filtered.csv"
TOP_K = 10
MAX_K = 10


# ═══════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════

def load_all_models(device):
    """加载 models/ 下所有 .pth，跳过不含 unit_embed 的旧架构"""
    import torch
    import train
    import importlib
    from train import UnitAwareTransformer

    # 注入 __main__ 以便 pickle 反序列化找到类（torch 仍在函数内导入，worker 不触发）
    import __main__
    __main__.UnitAwareTransformer = UnitAwareTransformer

    if not hasattr(train, "EnsembleModel"):
        train.EnsembleModel = UnitAwareTransformer
    if not hasattr(train, "UnitAwareTransformer"):
        train.UnitAwareTransformer = UnitAwareTransformer
    try:
        importlib.import_module("models")
    except ImportError:
        pass
    sys.modules["models.model"] = train
    sys.modules.setdefault("ensemble_model", train)

    model_files = sorted(MODEL_DIR.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"{MODEL_DIR} 下未找到 .pth 文件")

    models = {}
    skipped = []
    for f in model_files:
        try:
            m = torch.load(str(f), map_location=device, weights_only=False)
            if not hasattr(m, 'unit_embed'):
                raise RuntimeError("不兼容的旧架构（缺少 unit_embed）")
            m.eval()
            models[f.stem] = m
            print(f"  ✓ {f.stem}")
        except Exception as e:
            skipped.append(f"{f.name}: {e}")
    if skipped:
        print(f"\n  跳过 {len(skipped)} 个不兼容的旧模型:")
        for s in skipped:
            print(f"    ✗ {s}")
    return models


# ═══════════════════════════════════════════════════════════════
# 预测
# ═══════════════════════════════════════════════════════════════

def _predict_one_model(model, dataset, device, name, track_nan_sample=False):
    """单模型全量预测 — NaN/Inf 防御"""
    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)
    preds = []
    nan_row_indices = []
    nan_sample_info = None
    batch_offset = 0
    with torch.no_grad():
        for ls, lc, rs, rc, _ in loader:
            B = ls.shape[0]
            ls, lc, rs, rc = [x.to(device) for x in (ls, lc, rs, rc)]
            for t in (ls, lc, rs, rc):
                if torch.isnan(t).any():
                    t[torch.isnan(t)] = 0
                if torch.isinf(t).any():
                    t[torch.isinf(t)] = 0
            p = model(ls, lc, rs, rc).cpu().numpy()
            nan_mask = np.isnan(p)
            if nan_mask.any():
                local_idx = np.flatnonzero(nan_mask)
                nan_row_indices.extend((batch_offset + local_idx).tolist())
                if track_nan_sample and nan_sample_info is None:
                    gidx = batch_offset + int(local_idx[0])
                    lc_np = lc[local_idx[0]].cpu().numpy()
                    rc_np = rc[local_idx[0]].cpu().numpy()
                    nan_sample_info = (
                        gidx, float(lc_np.max()), float(rc_np.max()),
                        sorted(lc_np[lc_np > 0])[-5:].tolist() if (lc_np > 0).any() else [],
                        sorted(rc_np[rc_np > 0])[-5:].tolist() if (rc_np > 0).any() else [],
                    )
            p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
            preds.append(p)
            batch_offset += B
    return name, np.concatenate(preds), nan_row_indices, nan_sample_info


def get_all_predictions(models, dataset, device, desc="预测中"):
    """顺序全模型预测 — 单模型满载 GPU，低显存自动降 batch"""
    from tqdm import tqdm
    import torch

    batch_size = 2048 if device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3 else 4096
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_labels = []
    for _, _, _, _, labels in loader:
        all_labels.append(labels.cpu().numpy())
    labels_array = np.concatenate(all_labels)

    model_names = list(models.keys())
    results = {}
    all_nan_rows = set()
    nan_sample_info = None
    for i, name in enumerate(tqdm(model_names, desc=desc)):
        _, preds, nan_rows, info = _predict_one_model(
            models[name], dataset, device, name, track_nan_sample=(i == 0))
        results[name] = preds
        if nan_rows:
            all_nan_rows.update(nan_rows)
        if info:
            nan_sample_info = info

    pred_matrix = np.column_stack([results[name] for name in model_names])
    return pred_matrix, labels_array, model_names, all_nan_rows, nan_sample_info


def evaluate_combination(preds_subset, labels, weights):
    """加权集成 → 准确率"""
    weighted = preds_subset @ weights
    return ((weighted > 0.5).astype(float) == labels).mean()


# ═══════════════════════════════════════════════════════════════
# 多进程 LR 拟合
# ═══════════════════════════════════════════════════════════════

_worker_data = None


def _init_worker(X_full, y_full, cw):
    """进程初始化 — 设环境 + 存数据（只执行一次）"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    global _worker_data
    _worker_data = (X_full, y_full, cw)


def _eval_one_combo(combo_idx):
    """LR 拟合权重 — 从全局取数据"""
    X_full, y_full, cw = _worker_data
    idx = list(combo_idx)
    X_tr = X_full[:, idx]
    y_tr = y_full

    if np.isnan(X_tr).any():
        w = np.ones(len(idx)) / len(idx)
        return (combo_idx, w)

    for C, solver in [(1.0, 'lbfgs'), (1.0, 'saga')]:
        try:
            lr = LogisticRegression(
                C=C, solver=solver, max_iter=200,
                class_weight=cw
            )
            lr.fit(X_tr, y_tr)
            w = lr.coef_[0]
            w = w / (np.abs(w).sum() + 1e-10)
            return (combo_idx, w)
        except Exception:
            continue
    w = np.ones(len(idx)) / len(idx)
    return (combo_idx, w)


# ═══════════════════════════════════════════════════════════════
# 辅助：自适应进程数
# ═══════════════════════════════════════════════════════════════

def _estimate_worker_count():
    """根据可用内存估算安全进程数（每进程约 200MB）"""
    import psutil
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    cpu_count = os.cpu_count() or 4
    mem_limit = max(2, int(avail_gb / 0.25))  # 每进程预算 250MB
    return min(cpu_count, 16, mem_limit)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    from itertools import combinations as combo_iter
    from tqdm import tqdm
    from multiprocessing import Pool as MPool

    import torch
    from train import UnitAwareTransformer, ArknightsDataset, get_device

    device = get_device()
    print(f"设备: {device}")

    if device.type == "cuda":
        free_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU 显存: {free_mem:.1f} GB")

    # 1. 加载模型
    print(f"\n加载模型 ({MODEL_DIR}):")
    models = load_all_models(device)
    n_models = len(models)
    if n_models < 2:
        print("  错误: 至少需要 2 个模型才能搜索组合")
        return
    print(f"共 {n_models} 个模型\n")

    # 2. 加载数据
    print(f"训练集: {TRAIN_FILE}")
    dataset_train = ArknightsDataset(TRAIN_FILE, max_value=100)
    print(f"  样本数: {len(dataset_train)}")
    print(f"验证集: {VAL_FILE}")
    dataset_val = ArknightsDataset(VAL_FILE, max_value=100)
    print(f"  样本数: {len(dataset_val)}")

    # 3. GPU 预测
    print("\n训练集预测中...")
    pred_train, labels_train, model_names, all_nan_rows, _ = get_all_predictions(
        models, dataset_train, device, desc="训练集")
    if all_nan_rows:
        print(f"  剔除 {len(all_nan_rows)} 条异常样本", flush=True)
        keep_mask = np.ones(pred_train.shape[0], dtype=bool)
        keep_mask[list(all_nan_rows)] = False
        pred_train = pred_train[keep_mask]
        labels_train = labels_train[keep_mask]
    print(f"  预测矩阵: {pred_train.shape}")

    print("\n验证集预测中...")
    pred_val, labels_val, _, all_nan_rows_val, _ = get_all_predictions(
        models, dataset_val, device, desc="验证集")
    if all_nan_rows_val:
        keep_mask_val = np.ones(pred_val.shape[0], dtype=bool)
        keep_mask_val[list(all_nan_rows_val)] = False
        pred_val = pred_val[keep_mask_val]
        labels_val = labels_val[keep_mask_val]
        print(f"  剔除 {len(all_nan_rows_val)} 条异常样本", flush=True)
    print(f"  预测矩阵: {pred_val.shape}")

    # 4. 单模型基线
    print("\n单模型基线（验证集准确率）:")
    baselines = {}
    for i, name in enumerate(model_names):
        acc = ((pred_val[:, i] > 0.5).astype(float) == labels_val).mean()
        baselines[name] = acc
    for rank, (name, acc) in enumerate(sorted(baselines.items(), key=lambda x: -x[1]), 1):
        print(f"  {rank:>2}. {name[:50]:<50} {acc:.4f}")

    # 5. 等权重基线
    print("\n等权重集成基线（k=1~全部）:")
    eq_baselines = {}
    for k in range(1, n_models + 1):
        top_k_idx = sorted(baselines.keys(), key=lambda x: -baselines[x])[:k]
        top_k_pred = pred_val[:, [model_names.index(n) for n in top_k_idx]]
        eq_acc = ((top_k_pred.mean(axis=1) > 0.5).astype(float) == labels_val).mean()
        eq_baselines[k] = eq_acc
        print(f"  k={k:>2}: 等权={eq_acc:.4f}")
    best_eq_k = max(eq_baselines, key=eq_baselines.get)
    print(f"  等权最佳: k={best_eq_k}, acc={eq_baselines[best_eq_k]:.4f}")
    if best_eq_k > MAX_K:
        print(f"  ⚠ 等权最佳 k={best_eq_k} > MAX_K={MAX_K}，考虑放宽截断")

    # 6. 数据均衡
    pos_ratio = labels_train.mean()
    print(f"\n训练集正负比: {pos_ratio:.3f} (1=全正, 0=全负)")
    if 0.40 <= pos_ratio <= 0.60:
        print("  正负均衡 → 使用无偏LR")
        use_balanced = None
    else:
        print("  正负不均衡 → 使用 class_weight='balanced'")
        use_balanced = 'balanced'

    # 7. 生成所有组合
    max_k = min(n_models, MAX_K)
    all_combos = []
    for k in range(2, max_k + 1):
        all_combos.extend((combo, k) for combo in combo_iter(range(n_models), k))
    total_combos = len(all_combos)
    full_total = sum(1 for k in range(2, n_models + 1) for _ in combo_iter(range(n_models), k))
    print(f"\n全部组合: {full_total:,} 种 → 截断至 k≤{max_k}: {total_combos:,} 种 "
          f"({total_combos / full_total * 100:.0f}%)")

    # 8. 并行 LR 拟合（增量缓存 — 模型名做 key）
    import hashlib
    import pickle
    import gzip

    try:
        n_workers = _estimate_worker_count()
    except Exception:
        n_workers = min(os.cpu_count() or 4, 8)

    combo_inputs = [c for c, _ in all_combos]

    def _combo_name_key(combo_idx):
        return tuple(sorted([model_names[i] for i in combo_idx]))

    # 数据指纹 — 不匹配则全部失效
    data_fp = hashlib.md5(
        pred_train.tobytes() + labels_train.tobytes()
        + str(use_balanced).encode()
    ).hexdigest()[:12]

    cache_dir = MODEL_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "lr_cache.pkl.gz"
    cache_weights = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rb") as f:
            saved = pickle.load(f)
        if isinstance(saved, dict) and saved.get("_data_fp") == data_fp:
            cache_weights = saved.get("weights", {})
        else:
            print("  缓存失效（数据已变），将重新拟合")

    # 分离已缓存和待计算
    lr_results = [None] * len(combo_inputs)
    missing = []
    cached_count = 0
    for i, combo_idx in enumerate(combo_inputs):
        if _combo_name_key(combo_idx) in cache_weights:
            lr_results[i] = (combo_idx, cache_weights[_combo_name_key(combo_idx)])
            cached_count += 1
        else:
            missing.append(combo_idx)

    if missing:
        if cached_count:
            print(f"  ✓ 缓存命中 {cached_count}/{total_combos} 组合 → 只算 {len(missing)} 个新组合")
        chunksz = max(1, len(missing) // (n_workers * 3))
        print(f"多进程并行拟合 LR ({n_workers} 进程, chunksize={chunksz})...")
        combo_to_index = {c: i for i, c in enumerate(combo_inputs)}
        with MPool(processes=n_workers,
                   initializer=_init_worker,
                   initargs=(pred_train, labels_train, use_balanced)) as pool:
            for combo_idx, w in tqdm(
                pool.imap_unordered(_eval_one_combo, missing, chunksize=chunksz),
                total=len(missing), desc="LR拟合", mininterval=0.2,
                file=sys.stdout, ncols=80):
                cache_weights[_combo_name_key(combo_idx)] = w
                lr_results[combo_to_index[combo_idx]] = (combo_idx, w)
        with gzip.open(cache_path, "wb") as f:
            pickle.dump({"_data_fp": data_fp, "weights": cache_weights},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  缓存已更新: {cache_path.name} ({len(cache_weights)} 条目)")
    else:
        print(f"  ✓ 全部命中缓存 {cache_path.name}，跳过 LR 拟合")

    # 9. 评估
    print("\n评估组合中...")
    results_by_k = {}
    for (combo, k), (idx_list, weights) in tqdm(
        zip(all_combos, lr_results), total=total_combos,
        desc="评估", mininterval=0.3, file=sys.stdout, ncols=80):
        acc = evaluate_combination(pred_val[:, list(idx_list)], labels_val, weights)
        results_by_k.setdefault(k, []).append((idx_list, weights, acc))

    all_results = []
    for k in range(2, max_k + 1):
        results_by_k[k].sort(key=lambda x: -x[2])
        best_acc = results_by_k[k][0][2]
        print(f"  k={k:>2}: 最佳 {best_acc:.4f}")
        for idx_list, weights, acc in results_by_k[k][:5]:
            combo_names = [model_names[i] for i in idx_list]
            all_results.append({'models': combo_names, 'weights': weights, 'acc': acc, 'k': k})

    all_results.sort(key=lambda x: -x['acc'])

    print(f"\n{'=' * 90}")
    print(f"  最佳 {TOP_K} 组模型 — 验证集 {len(labels_val)} 条")
    print(f"{'=' * 90}")
    print(f"{'排名':<5}{'准确率':>8}{'模型数':>6}  {'模型组合':<60}")
    print(f"{'-' * 90}")
    for rank, r in enumerate(all_results[:TOP_K], 1):
        model_str = " + ".join([n[:28] for n in r['models']])
        if len(model_str) > 55:
            model_str = model_str[:52] + "..."
        print(f"{rank:<5}{r['acc']:>8.4f}{r['k']:>6}  {model_str}")

    best = all_results[0]
    print(f"\n最佳权重:")
    for name, w in zip(best['models'], best['weights']):
        print(f"  {name[:50]:<50} {w:+.4f}")

    best_single = max(baselines.values())
    print(f"\n单模型最佳: {best_single:.4f} → 集成最佳: {best['acc']:.4f} "
          f"(+{best['acc'] - best_single:+.4f})")

    # 10. 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = MODEL_DIR / f"model_selection_{ts}.csv"
    rows = []
    for rank, r in enumerate(all_results[:TOP_K], 1):
        rows.append({
            '排名': rank,
            '准确率': f"{r['acc']:.4f}",
            '模型数': r['k'],
            '模型组合': " + ".join(r['models']),
            '权重': ", ".join([f"{w:+.4f}" for w in r['weights']]),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {out_path}")

    import json
    weights_path = MODEL_DIR / "model_weights.json"
    weights_json = {name: round(float(w), 4)
                    for name, w in zip(best['models'], best['weights'])}
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights_json, f, ensure_ascii=False, indent=2)
    print(f"权重文件已同步: {weights_path}")
    if any(w == 0 for w in weights_json.values()):
        print(f"  排除: {[n[:40] for n, w in weights_json.items() if w == 0]}")


if __name__ == "__main__":
    main()
