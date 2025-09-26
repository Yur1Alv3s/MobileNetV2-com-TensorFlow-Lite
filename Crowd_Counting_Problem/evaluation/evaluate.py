# evaluation/evaluate.py
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from utils.paths import ARTIFACTS, load_cfg
from data.datasets import iter_ids, load_pair


# ----------------------- utilitários de métrica -----------------------

def _counts_from_maps(maps: np.ndarray | tf.Tensor) -> np.ndarray:
    """
    _counts_from_maps(maps) -> np.ndarray

    Converte mapas de densidade [H,W,1] (ou batelada [B,H,W,1]) em contagens pela soma dos pixels.
    Entrada:
      - maps: array/tensor float com shape [..., H, W, 1]
    Saída:
      - np.ndarray 1D com as contagens por item do batch.
    """
    if isinstance(maps, tf.Tensor):
        maps = maps.numpy()
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim == 3:
        maps = maps[None, ...]
    return maps.sum(axis=(1, 2, 3))


def _mae_rmse(gt: List[float], pred: List[float]) -> Tuple[float, float]:
    """
    _mae_rmse(gt: list[float], pred: list[float]) -> (mae, rmse)

    Calcula MAE e RMSE entre duas listas de contagens.
    Entrada:
      - gt: lista com contagens verdadeiras.
      - pred: lista com contagens previstas.
    Saída:
      - (MAE, RMSE) como floats.
    """
    assert len(gt) == len(pred)
    errs = [abs(a - b) for a, b in zip(gt, pred)]
    mae = mean(errs)
    rmse = (mean([(a - b) ** 2 for a, b in zip(gt, pred)]) ** 0.5)
    return float(mae), float(rmse)


# ----------------------------- KERAS ---------------------------------

def evaluate_keras(model_path: str | Path, split: str, cfg: dict, limit: Optional[int] = None) -> Dict:
    """
    evaluate_keras(model_path, split, cfg, limit=None) -> dict

    Avalia um modelo Keras (.keras) em um split (train/val), computando MAE/RMSE de contagem
    e salvando um CSV por-imagem.

    Entrada:
      - model_path: caminho do .keras treinado.
      - split: "train" ou "val".
      - cfg: dicionário de config (usa input_size/output_stride).
      - limit: se informado, avalia apenas as primeiras 'limit' imagens.
    Saída:
      - dict: {"MAE": float, "RMSE": float, "n": int, "csv_path": str}
    Efeitos colaterais:
      - Salva CSV em artifacts/metrics/<exp>_keras_<split>.csv com colunas:
        id, gt_count, pred_count, abs_err
    """
    model_path = Path(model_path)
    assert model_path.exists(), f".keras não encontrado: {model_path}"

    exp = cfg.get("experiment_name", "crowd_mnv2_s8")
    out_csv = ARTIFACTS / "metrics" / f"{exp}_keras_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Carrega o modelo sem compilar (só inferência)
    model = tf.keras.models.load_model(str(model_path), compile=False)

    ids = iter_ids(split)
    if limit is not None:
        ids = ids[:int(limit)]

    gts: List[float] = []
    preds: List[float] = []

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "gt_count", "pred_count", "abs_err"])

        for i in ids:
            img, den = load_pair(i, split, cfg)  # img [H,W,3] em [-1,1]; den [Hs,Ws,1]
            gt = float(den.sum())
            # inferência
            y = model(np.expand_dims(img, 0), training=False)
            pred = float(_counts_from_maps(y)[0])

            gts.append(gt)
            preds.append(pred)
            w.writerow([i, gt, pred, abs(gt - pred)])

    mae, rmse = _mae_rmse(gts, preds)
    return {"MAE": mae, "RMSE": rmse, "n": len(ids), "csv_path": str(out_csv)}


# ----------------------------- TFLITE --------------------------------

def _load_tflite_interpreter(tflite_path: str | Path, delegate: str = "cpu") -> tf.lite.Interpreter:
    """
    _load_tflite_interpreter(tflite_path, delegate='cpu') -> Interpreter

    Cria um Interpreter do TFLite com o delegate escolhido.
    Entrada:
      - tflite_path: caminho do arquivo .tflite.
      - delegate: "cpu" | "gpu" | "nnapi" (pode variar por plataforma).
    Saída:
      - tf.lite.Interpreter pronto para uso (allocate_tensors já chamado).
    """
    tflite_path = str(tflite_path)
    if delegate.lower() == "gpu":
        try:
            from tensorflow.lite.experimental import Interpreter as GPUInterpreter  # type: ignore
            interpreter = GPUInterpreter(model_path=tflite_path, experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')])  # noqa: E501
        except Exception:
            # Fallback para CPU se GPU delegate não estiver disponível
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
    elif delegate.lower() == "nnapi":
        try:
            interpreter = tf.lite.Interpreter(
                model_path=tflite_path,
                experimental_delegates=[tf.lite.experimental.load_delegate('libnnapi_delegate.so')]
            )
        except Exception:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
    else:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)

    interpreter.allocate_tensors()
    return interpreter


def _tflite_predict_count(interpreter: tf.lite.Interpreter, img: np.ndarray) -> float:
    """
    _tflite_predict_count(interpreter, img) -> float

    Roda uma imagem [H,W,3] em [-1,1] no TFLite e devolve a **contagem** (soma do mapa).
    Entrada:
      - interpreter: já alocado.
      - img: np.ndarray float32 [H,W,3] em [-1,1].
    Saída:
      - Contagem prevista (float).
    Observações:
      - Se o modelo for INT8, fazemos o de/para int8 automaticamente usando scale/zero_point do tensor.
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = np.expand_dims(img, axis=0)

    if input_details["dtype"] == np.int8:
        # quantização assimétrica: x_int8 = x_float/scale + zero_point
        scale, zero = input_details["quantization"]
        x = (x / scale + zero).astype(np.int8)

    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()

    y = interpreter.get_tensor(output_details["index"])

    if output_details["dtype"] == np.int8:
        scale, zero = output_details["quantization"]
        y = (y.astype(np.float32) - zero) * scale  # volta p/ float

    return float(_counts_from_maps(y)[0])


def evaluate_tflite(tflite_path: str | Path, split: str, cfg: dict, delegate: str = "cpu", limit: Optional[int] = None) -> Dict:
    """
    evaluate_tflite(tflite_path, split, cfg, delegate="cpu", limit=None) -> dict

    Avalia um modelo TFLite (.tflite) em um split, computando MAE/RMSE de contagem
    e salvando um CSV por-imagem.

    Entrada:
      - tflite_path: caminho para .tflite.
      - split: "train" ou "val".
      - cfg: dicionário de config (usa input_size/output_stride).
      - delegate: "cpu" | "gpu" | "nnapi".
      - limit: se informado, avalia apenas as primeiras 'limit' imagens.
    Saída:
      - dict: {"MAE": float, "RMSE": float, "n": int, "csv_path": str}
    Efeitos colaterais:
      - Salva CSV em artifacts/metrics/<exp>_<mode>_<delegate>_<split>.csv
    """
    tflite_path = Path(tflite_path)
    assert tflite_path.exists(), f".tflite não encontrado: {tflite_path}"

    exp = cfg.get("experiment_name", "crowd_mnv2_s8")
    # tenta inferir o "mode" a partir do nome do arquivo
    mode = "fp32"
    name_lower = tflite_path.name.lower()
    if "fp16" in name_lower:
        mode = "fp16"
    elif "int8" in name_lower:
        mode = "int8"

    out_csv = ARTIFACTS / "metrics" / f"{exp}_{mode}_{delegate}_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    interpreter = _load_tflite_interpreter(tflite_path, delegate=delegate)

    ids = iter_ids(split)
    if limit is not None:
        ids = ids[:int(limit)]

    gts: List[float] = []
    preds: List[float] = []

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "gt_count", "pred_count", "abs_err"])

        for i in ids:
            img, den = load_pair(i, split, cfg)
            gt = float(den.sum())
            pred = _tflite_predict_count(interpreter, img)

            gts.append(gt)
            preds.append(pred)
            w.writerow([i, gt, pred, abs(gt - pred)])

    mae, rmse = _mae_rmse(gts, preds)
    return {"MAE": mae, "RMSE": rmse, "n": len(ids), "csv_path": str(out_csv)}


# ----------------------- COMPARAÇÃO DIRETA ---------------------------

def compare_keras_vs_tflite(model_path: str | Path, tflite_path: str | Path, split: str, cfg: dict, limit: Optional[int] = None) -> Dict:
    """
    compare_keras_vs_tflite(model_path, tflite_path, split, cfg, limit=None) -> dict

    Compara **contagens** do Keras vs TFLite na MESMA lista de imagens,
    medindo o delta médio e percentis — útil para validar a conversão.

    Entrada:
      - model_path: .keras (Keras FP32).
      - tflite_path: .tflite (FP16/INT8).
      - split: "train" ou "val".
      - cfg: dicionário de config.
      - limit: avalia só as primeiras 'limit' imagens se informado.
    Saída:
      - dict com resumo, ex.:
        {
          "mean_abs_delta": float,
          "p95_abs_delta":  float,
          "n": int,
          "csv_path": "artifacts/metrics/<exp>_compare_keras_tflite_<split>.csv"
        }
    Efeitos colaterais:
      - CSV com colunas: id, count_keras, count_tflite, abs_delta
    """
    model = tf.keras.models.load_model(str(model_path), compile=False)
    interpreter = _load_tflite_interpreter(tflite_path, delegate="cpu")

    ids = iter_ids(split)
    if limit is not None:
        ids = ids[:int(limit)]

    rows = []
    deltas = []

    for i in ids:
        img, _ = load_pair(i, split, cfg)
        # Keras
        yk = model(np.expand_dims(img, 0), training=False)
        ck = float(_counts_from_maps(yk)[0])
        # TFLite
        ct = _tflite_predict_count(interpreter, img)

        d = abs(ck - ct)
        deltas.append(d)
        rows.append([i, ck, ct, d])

    exp = cfg.get("experiment_name", "crowd_mnv2_s8")
    out_csv = ARTIFACTS / "metrics" / f"{exp}_compare_keras_tflite_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "count_keras", "count_tflite", "abs_delta"])
        w.writerows(rows)

    deltas_sorted = sorted(deltas)
    p95 = deltas_sorted[int(0.95 * (len(deltas_sorted) - 1))] if deltas_sorted else 0.0
    mean_abs_delta = mean(deltas) if deltas else 0.0

    return {
        "mean_abs_delta": float(mean_abs_delta),
        "p95_abs_delta": float(p95),
        "n": len(ids),
        "csv_path": str(out_csv),
    }


if __name__ == "__main__":
    # Exemplos de uso:
    #  python -m evaluation.evaluate --keras artifacts/models/crowd_mnv2_s8.keras --split val
    #  python -m evaluation.evaluate --tflite artifacts/models/crowd_mnv2_s8_fp16.tflite --split val --delegate cpu
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--keras", type=str, default=None, help="caminho do .keras (para avaliar Keras)")
    parser.add_argument("--tflite", type=str, default=None, help="caminho do .tflite (para avaliar TFLite)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--delegate", type=str, default="cpu", choices=["cpu", "gpu", "nnapi"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config) if args.config else load_cfg()

    if args.keras and not args.tflite:
        print(evaluate_keras(args.keras, args.split, cfg, limit=args.limit))
    elif args.tflite and not args.keras:
        print(evaluate_tflite(args.tflite, args.split, cfg, delegate=args.delegate, limit=args.limit))
    elif args.keras and args.tflite:
        print(compare_keras_vs_tflite(args.keras, args.tflite, args.split, cfg, limit=args.limit))
    else:
        parser.print_help()
