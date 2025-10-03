#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação comparativa de modelos Keras (.keras/.h5) e TFLite (.tflite)
para tarefas de Classificação Binária ou Regressão, usando o MESMO
conjunto de teste e reportando métricas, tamanhos e latências
(p50/p95/p99) com aquecimento (warm-up).

>>> NOVO <<<
- A medição de memória pico (RSS) foi REMOVIDA de `evaluate_models()`
  e movida para funções separadas para rodar em outra execução:
  * `measure_peak_rss_keras(...)`
  * `measure_peak_rss_tflite(...)`

Requisitos: numpy, tensorflow (ou tflite_runtime opcional para TFLite), psutil (opcional, só nas funções de memória).
"""
from __future__ import annotations

import os
import time
import math
import json
import random
import platform
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Forçar CPU antes de importar TensorFlow (comparabilidade)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

try:
    import psutil  # usado apenas nas funções de memória
except Exception:
    psutil = None

try:
    import resource  # Unix
except Exception:
    resource = None


# =============================
# Utilitários de aleatoriedade
# =============================

def set_global_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


# =============================
# Utilitários de Dataset
# =============================
ArrayLike = Union[np.ndarray, "tf.Tensor"]
DatasetLike = Union[Tuple[ArrayLike, ArrayLike], "tf.data.Dataset"]


def _is_tf_dataset(obj: Any) -> bool:
    return hasattr(tf.data, "Dataset") and isinstance(obj, tf.data.Dataset)


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def dataset_iterator(test_data: DatasetLike) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Itera (X_batch, y_batch) como numpy, suportando:
       - tf.data.Dataset
       - tupla única (X, y)
       - lista de lotes [(Xb, yb), ...]
    """
    # Caso A: tf.data.Dataset
    if _is_tf_dataset(test_data):
        for batch in test_data:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                Xb, yb = batch[0], batch[1]
            else:
                raise ValueError("tf.data.Dataset deve produzir tuplas (X, y)")
            yield _to_numpy(Xb), _to_numpy(yb)
        return

    # Caso B: lista/tupla de lotes [(Xb, yb), (Xb, yb), ...]
    if isinstance(test_data, (list, tuple)) and len(test_data) > 0 \
       and isinstance(test_data[0], (list, tuple)) and len(test_data[0]) >= 2:
        for Xb, yb in test_data:
            yield _to_numpy(Xb), _to_numpy(yb)
        return

    # Caso C: tupla única (X, y)
    if isinstance(test_data, (tuple, list)) and len(test_data) >= 2:
        X, y = test_data
        Xn, yn = _to_numpy(X), _to_numpy(y)
        if Xn.ndim == 3:  # (H,W,C) -> (1,H,W,C)
            Xn = np.expand_dims(Xn, 0)
        if yn.ndim == 0:
            yn = np.expand_dims(yn, 0)
        if Xn.shape[0] != yn.shape[0]:
            raise ValueError(f"Batch mismatch: X tem {Xn.shape[0]} e y tem {yn.shape[0]}")
        bs = 32
        for i in range(0, Xn.shape[0], bs):
            yield Xn[i:i+bs], yn[i:i+bs]
        return

    raise ValueError("Formato de test_data não suportado: use tf.data.Dataset, (X, y) ou lista de (Xb, yb).")


def materialize_to_list(test_data: DatasetLike, limit_samples: Optional[int] = None):
    cached = []
    n = 0
    for Xb, yb in dataset_iterator(test_data):
        cached.append((Xb, yb))  # sem copy para evitar custo extra
        n += Xb.shape[0]
        if limit_samples is not None and n >= limit_samples:
            break
    return cached


def get_one_sample(test_data: DatasetLike, input_size: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    H, W, C = input_size
    for Xb, yb in dataset_iterator(test_data):
        x = Xb[0]
        y = yb[0]
        x = np.asarray(x)
        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        # garantir shape (1,H,W,C) se possível
        if x.shape[1:] != (H, W, C):
            try:
                # redimensiona via TF (assume imagem)
                x_res = tf.image.resize(x, (H, W)).numpy()
                if x_res.ndim == 3:
                    x_res = np.expand_dims(x_res, 0)
                if x_res.shape[-1] != C:
                    # grayscale -> 3 canais
                    if x_res.shape[-1] == 1 and C == 3:
                        x_res = np.repeat(x_res, 3, axis=-1)
                x = x_res
            except Exception:
                pass
        return x.astype(np.float32), np.asarray([y])
    raise ValueError("Dataset de teste está vazio.")


# =============================
# TFLite helpers
# =============================

class TFLiteWrapper:
    def __init__(self, model_path: str, num_threads: int = 1):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != 1 or len(self.output_details) != 1:
            raise ValueError("Este wrapper suporta apenas um tensor de entrada e um de saída.")

    def _quantize_if_needed(self, x: np.ndarray) -> np.ndarray:
        inp = self.input_details[0]
        dtype = inp["dtype"]
        if dtype == np.float32:
            return x.astype(np.float32)
        # quantizado (int8/uint8)
        scale, zero_point = inp["quantization"]
        if scale == 0:
            return x.astype(dtype)
        x_q = np.round(x / scale + zero_point).astype(dtype)
        return x_q

    def _dequantize_if_needed(self, yq: np.ndarray) -> np.ndarray:
        out = self.output_details[0]
        dtype = out["dtype"]
        if dtype == np.float32:
            return yq.astype(np.float32)
        scale, zero_point = out["quantization"]
        if scale == 0:
            return yq.astype(np.float32)
        y = (yq.astype(np.float32) - zero_point) * scale
        return y

    def predict_one(self, x1: np.ndarray) -> np.ndarray:
        # Ajustar shape se preciso (batch=1)
        inp_idx = self.input_details[0]["index"]
        out_idx = self.output_details[0]["index"]
        target_shape = tuple(self.input_details[0]["shape"])
        if target_shape[0] not in (1, -1):
            try:
                self.interpreter.resize_tensor_input(inp_idx, [1] + list(target_shape[1:]))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                out_idx = self.output_details[0]["index"]
            except Exception:
                pass
        xq = self._quantize_if_needed(x1)
        self.interpreter.set_tensor(inp_idx, xq)
        self.interpreter.invoke()
        yq = self.interpreter.get_tensor(out_idx)
        y = self._dequantize_if_needed(yq)
        return y


# =============================
# Métricas
# =============================

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(np.int32).ravel()
    probs = y_prob.astype(np.float32).ravel()
    # se não estiver em [0,1], aplicar sigmoid como fallback
    if probs.min() < 0.0 or probs.max() > 1.0:
        probs = sigmoid(probs)
    y_pred = (probs >= threshold).astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = max(1, len(y_true))
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float32).ravel()
    y_pred = y_pred.astype(np.float32).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


# =============================
# Predição em lote (Keras) e amostra-a-amostra (TFLite)
# =============================

def _counts_from_batch(arr: np.ndarray) -> np.ndarray:
    # arr: (B, ...) -> soma por amostra
    a = np.asarray(arr)
    if a.ndim == 1:
        return a
    return a.reshape(a.shape[0], -1).sum(axis=1)


def collect_predictions_keras(model: tf.keras.Model,
                              test_data: DatasetLike,
                              task_type: str,
                              batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (y_true, y_pred_prob_or_value)."""
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []

    for Xb, yb in dataset_iterator(test_data):
        Xb = Xb if Xb.ndim == 4 else np.expand_dims(Xb, 0)
        y_hat = model.predict(Xb, batch_size=min(batch_size, Xb.shape[0]), verbose=0)
        y_hat = np.asarray(y_hat)
        if task_type == "regressao":
            y_hat = _counts_from_batch(y_hat)
            yb_flat = _counts_from_batch(yb)
        else:
            if y_hat.ndim > 1:
                y_hat = y_hat.reshape((-1,))
            yb_flat = yb.reshape((-1,))
        y_true_list.append(yb_flat)
        y_pred_list.append(y_hat)

    y_true = np.concatenate(y_true_list, axis=0) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list, axis=0) if y_pred_list else np.array([])
    return y_true, y_pred


def collect_predictions_tflite(tfl: TFLiteWrapper,
                               test_data: DatasetLike,
                               task_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Predição amostra-a-amostra para evitar reallocate_tensors repetido.
    Retorna (y_true, y_pred_prob_or_value)."""
    y_true_list: List[float] = []
    y_pred_list: List[float] = []

    for Xb, yb in dataset_iterator(test_data):
        for i in range(Xb.shape[0]):
            x1 = Xb[i:i+1]
            y1 = yb[i]
            y_hat = tfl.predict_one(x1)
            if task_type == "regressao":
                y_pred_list.append(float(np.sum(y_hat)))
                y_true_list.append(float(np.sum(y1)))
            else:
                if y_hat.ndim > 1:
                    y_hat = y_hat.reshape((-1,))
                y_pred_list.append(float(y_hat[0]))
                y_true_list.append(float(np.ravel(y1)[0]))

    y_true = np.asarray(y_true_list, dtype=np.float32)
    y_pred = np.asarray(y_pred_list, dtype=np.float32)
    return y_true, y_pred


# =============================
# Latência (batch=1)
# =============================

def measure_latency_keras(model: tf.keras.Model,
                          sample_1x: np.ndarray,
                          warmup: int = 20,
                          runs: int = 100) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        _ = model.predict(sample_1x, batch_size=1, verbose=0)
    times = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        _ = model.predict(sample_1x, batch_size=1, verbose=0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
    }


def measure_latency_tflite(tfl: TFLiteWrapper,
                           sample_1x: np.ndarray,
                           warmup: int = 20,
                           runs: int = 100) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        _ = tfl.predict_one(sample_1x)
    times = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        _ = tfl.predict_one(sample_1x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
    }


# =============================
# Tamanho de arquivos
# =============================

def file_size_mb(path: str) -> float:
    return float(os.path.getsize(path)) / (1024 * 1024)


# =============================
# Função principal (SEM memória pico)
# =============================

def evaluate_models(
    task_type: str,
    keras_model_path: str,
    tflite_model_path: str,
    test_data: DatasetLike,
    input_size: Tuple[int, int, int] = (224, 224, 3),
    threshold: float = 0.5,
    latency_warmup: int = 20,
    latency_runs: int = 100,
    num_threads: int = 1,
    limit_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Avalia modelos Keras e TFLite no mesmo dataset.

    Retorna dicionário com MÉTRICAS, LATÊNCIA e TAMANHOS (sem memória pico).
    """
    assert task_type in {"classificacao", "regressao"}, "task_type deve ser 'classificacao' ou 'regressao'"

    set_global_seeds(42)

    # Preparar dataset (limitar amostras se solicitado)
    if limit_samples is not None and _is_tf_dataset(test_data):
        test_data = test_data.take(int(math.ceil(limit_samples / 1)))

    # Carregar modelos
    keras_model = tf.keras.models.load_model(keras_model_path, compile=False)
    tfl = TFLiteWrapper(tflite_model_path, num_threads=num_threads)

    # Tamanhos
    size_keras_mb = file_size_mb(keras_model_path)
    size_tflite_mb = file_size_mb(tflite_model_path)
    size_reduction_pct = 100.0 * (1.0 - (size_tflite_mb / size_keras_mb)) if size_keras_mb > 0 else 0.0

    # Coletar previsões
    cached = materialize_to_list(test_data, limit_samples=limit_samples)
    y_true_k, y_pred_k = collect_predictions_keras(keras_model, cached, task_type)
    y_true_t, y_pred_t = collect_predictions_tflite(tfl, cached, task_type)

    # Alinhar tamanhos se necessário
    if y_true_k.size != y_true_t.size:
        n = min(y_true_k.size, y_true_t.size)
        y_true_k, y_pred_k = y_true_k[:n], y_pred_k[:n]
        y_true_t, y_pred_t = y_true_t[:n], y_pred_t[:n]

    # Métricas
    if task_type == "classificacao":
        metrics_k = classification_metrics(y_true_k, y_pred_k, threshold=threshold)
        metrics_t = classification_metrics(y_true_t, y_pred_t, threshold=threshold)
    else:
        metrics_k = regression_metrics(y_true_k, y_pred_k)
        metrics_t = regression_metrics(y_true_t, y_pred_t)

    # Latências (batch=1)
    sample_1x, _ = get_one_sample(cached, input_size)
    lat_k = measure_latency_keras(keras_model, sample_1x, warmup=latency_warmup, runs=latency_runs)
    lat_t = measure_latency_tflite(tfl, sample_1x, warmup=latency_warmup, runs=latency_runs)

    # Metadados
    num_samples = int(min(len(y_true_k), len(y_true_t)))
    env = {
        "device": "cpu",
        "threads_tflite": int(num_threads),
        "tf_version": tf.__version__,
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
    }

    result = {
        "task": task_type,
        "dataset": {"num_samples": num_samples, "input_size": list(input_size), "batch_latency": 1},
        "env": env,
        "models": {
            "keras": {
                "size_mb": round(size_keras_mb, 3),
                "latency_ms": {k: round(v, 3) for k, v in lat_k.items()},
                "metrics": {k: round(v, 6) for k, v in metrics_k.items()},
            },
            "tflite": {
                "size_mb": round(size_tflite_mb, 3),
                "latency_ms": {k: round(v, 3) for k, v in lat_t.items()},
                "metrics": {k: round(v, 6) for k, v in metrics_t.items()},
            },
        },
        "size_reduction_pct": round(size_reduction_pct, 2),
    }

    # Impressão resumida
    print("=== Avaliação Comparativa (Keras vs TFLite) — SEM memória pico ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    def fmt_metrics(m: Dict[str, float]) -> str:
        return ", ".join([f"{k}={v:.4f}" for k, v in m.items()])

    print("Resumo:")
    print(f"Samples: {num_samples} | Input: {input_size} | TF {env['tf_version']} (CPU) | Threads TFLite: {num_threads}")
    print(f"Tamanho: Keras={size_keras_mb:.2f} MB | TFLite={size_tflite_mb:.2f} MB | Redução={result['size_reduction_pct']:.2f}%")
    print(
        f"Latência (ms): Keras p50={lat_k['p50']:.2f} p95={lat_k['p95']:.2f} p99={lat_k['p99']:.2f} | "
        f"TFLite p50={lat_t['p50']:.2f} p95={lat_t['p95']:.2f} p99={lat_t['p99']:.2f}"
    )
    if task_type == "classificacao":
        print("Métricas Keras:", fmt_metrics(metrics_k))
        print("Métricas TFLite:", fmt_metrics(metrics_t))
    else:
        print("Métricas Keras:", fmt_metrics(metrics_k))
        print("Métricas TFLite:", fmt_metrics(metrics_t))

    return result


# =============================
# Medição de MEMÓRIA (funções separadas)
#   → chame em OUTRA execução para evitar interferência
# =============================

def _get_rss_mb() -> float:
    """RSS atual do processo em MB (psutil preferido; fallback em resource)."""
    if psutil is not None:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    if resource is not None and hasattr(resource, "getrusage"):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        val_kb = getattr(usage, "ru_maxrss", 0)
        if val_kb <= 0:
            return 0.0
        if val_kb > 1024 * 1024:  # heurística: já em MB
            return float(val_kb)
        return float(val_kb) / 1024.0
    return 0.0


def measure_peak_rss_keras(
    task_type: str,
    keras_model_path: str,
    test_data: DatasetLike,
    batch_size: int = 32,
    limit_samples: Optional[int] = None,
) -> float:
    """Mede o pico de RSS executando inferência Keras sobre `test_data`.
    Execute em um processo separado do TFLite para isolar o pico.
    """
    set_global_seeds(42)
    model = tf.keras.models.load_model(keras_model_path, compile=False)
    cached = materialize_to_list(test_data, limit_samples=limit_samples)
    peak = 0.0
    for Xb, yb in dataset_iterator(cached):
        Xb = Xb if Xb.ndim == 4 else np.expand_dims(Xb, 0)
        y_hat = model.predict(Xb, batch_size=min(batch_size, Xb.shape[0]), verbose=0)
        if task_type == "regressao":
            _ = _counts_from_batch(y_hat)
        else:
            _ = y_hat.reshape((-1,)) if np.ndim(y_hat) > 1 else y_hat
        peak = max(peak, _get_rss_mb())
    return float(peak)


def measure_peak_rss_tflite(
    task_type: str,
    tflite_model_path: str,
    test_data: DatasetLike,
    num_threads: int = 1,
    limit_samples: Optional[int] = None,
) -> float:
    """Mede o pico de RSS executando inferência TFLite sobre `test_data`.
    Execute em um processo separado do Keras para isolar o pico.
    """
    set_global_seeds(42)
    tfl = TFLiteWrapper(tflite_model_path, num_threads=num_threads)
    cached = materialize_to_list(test_data, limit_samples=limit_samples)
    peak = 0.0
    for Xb, yb in dataset_iterator(cached):
        for i in range(Xb.shape[0]):
            x1 = Xb[i:i+1]
            y_hat = tfl.predict_one(x1)
            if task_type == "regressao":
                _ = float(np.sum(y_hat))
            else:
                _ = float(np.ravel(y_hat)[0])
            peak = max(peak, _get_rss_mb())
    return float(peak)


# =============================
# Exemplo de uso (comentado)
# =============================
if __name__ == "__main__":
    """
    Exemplos mínimos — ajuste os caminhos e o carregamento do dataset conforme seu projeto.

    # 1) Avaliação comparativa (sem memória pico)
    >>> res = evaluate_models(
    >>>     task_type="classificacao",
    >>>     keras_model_path="/caminho/modelo_cls.keras",
    >>>     tflite_model_path="/caminho/modelo_cls.tflite",
    >>>     test_data=(np.random.rand(64,224,224,3).astype(np.float32), np.random.randint(0,2,(64,))),
    >>>     input_size=(224,224,3),
    >>> )

    # 2) Medição de pico de memória — rode em EXECUÇÕES SEPARADAS
    >>> peak_k = measure_peak_rss_keras(
    >>>     task_type="classificacao",
    >>>     keras_model_path="/caminho/modelo_cls.keras",
    >>>     test_data=(np.random.rand(64,224,224,3).astype(np.float32), np.random.randint(0,2,(64,))),
    >>> )
    >>> peak_t = measure_peak_rss_tflite(
    >>>     task_type="classificacao",
    >>>     tflite_model_path="/caminho/modelo_cls.tflite",
    >>>     test_data=(np.random.rand(64,224,224,3).astype(np.float32), np.random.randint(0,2,(64,))),
    >>> )
    """
    pass
