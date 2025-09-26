# utils/model_info.py
from __future__ import annotations
from pathlib import Path
from typing import Union, Iterable, Dict, Any
import numpy as np
import tensorflow as tf
from collections import Counter
import time

PathLike = Union[str, Path]
KerasLike = Union[tf.keras.Model, PathLike]

# ------------------------------------------------------------
# Decorator resiliente para registrar objetos customizados
# (funciona em diferentes arranjos de Keras/TF; vira no-op se não houver)
# ------------------------------------------------------------
def _resolve_register():
    # 1) TF-Keras mais recente (pode não ter .saving)
    try:
        return tf.keras.saving.register_keras_serializable  # type: ignore[attr-defined]
    except Exception:
        pass
    # 2) Keras 3 (pacote 'keras')
    try:
        from keras.saving import register_keras_serializable as _reg  # type: ignore
        return _reg
    except Exception:
        pass
    # 3) TF-Keras: utils
    try:
        return tf.keras.utils.register_keras_serializable
    except Exception:
        pass
    # 4) Fallback: no-op
    def _noop_register(*args, **kwargs):
        def deco(fn): return fn
        return deco
    return _noop_register

_REGISTER = _resolve_register()

# Stub opcional para perdas/métricas customizadas
# (não será usado porque carregamos com compile=False, mas mantém robustez)
@_REGISTER()
def mae_count(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

_CUSTOM_OBJECTS = {"mae_count": mae_count}

# -------------------------------- Utils --------------------------------
def _human_size(n: int) -> str:
    u = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(u)-1:
        f /= 1024.0; i += 1
    return f"{f:.2f} {u[i]}"

def _file_size(p: Path) -> str:
    try:
        return _human_size(Path(p).stat().st_size)
    except Exception:
        return "N/A"

def _flatten_layers(layer) -> Iterable[tf.keras.layers.Layer]:
    if hasattr(layer, "layers") and layer.layers:
        for l in layer.layers:
            yield from _flatten_layers(l)
    else:
        yield layer

def _safe_dtype_name(d) -> str:
    try:
        return d.name  # tf.DType
    except Exception:
        return str(d)

# ------------------------------ KERAS -----------------------------------
def _safe_load_model(p: Path) -> tf.keras.Model:
    """
    Carregamento resiliente:
      - compile=False evita deserializar losses/metrics/optimizer
      - custom_objects inclui stubs seguros (ex.: mae_count) se necessário
    """
    try:
        return tf.keras.models.load_model(str(p), compile=False)
    except Exception:
        return tf.keras.models.load_model(str(p), compile=False, custom_objects=_CUSTOM_OBJECTS)

def _keras_io_shapes(model: tf.keras.Model):
    in_shapes = [tuple(x.shape) for x in getattr(model, "inputs", [])]
    out_shapes = [tuple(x.shape) for x in getattr(model, "outputs", [])]
    n_classes = None
    if out_shapes:
        last = out_shapes[0]
        if len(last) >= 2 and last[-1] is not None:
            try:
                n_classes = int(last[-1])
            except Exception:
                n_classes = None
    return in_shapes, out_shapes, n_classes

def _keras_layer_histogram(model: tf.keras.Model) -> Dict[str,int]:
    layers = list(_flatten_layers(model))
    kinds = [l.__class__.__name__ for l in layers]
    return dict(Counter(kinds).most_common())

def _analyze_keras(obj: KerasLike, details: bool) -> dict:
    if isinstance(obj, tf.keras.Model):
        model = obj
        src = getattr(model, "name", "keras_model")
        fsize = "N/A"
    else:
        p = Path(obj)
        src = str(p)
        model = _safe_load_model(p)
        fsize = _file_size(p)

    layers = list(_flatten_layers(model))
    num_layers = len(layers)
    dense_units = sum(getattr(l, "units", 0) for l in layers if isinstance(l, tf.keras.layers.Dense))

    # dtypes/precisão
    if getattr(model, "variables", None):
        var_dtypes = sorted({_safe_dtype_name(getattr(v, "dtype", "desconhecido")) for v in model.variables})
    else:
        var_dtypes = ["desconhecido"]
    try:
        policy = str(model.dtype_policy)
    except Exception:
        policy = "/".join(var_dtypes)

    # parâmetros
    try:
        trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
        non_trainable_params = int(np.sum([np.prod(v.shape) for v in model.non_trainable_variables]))
        total_params = int(model.count_params())
    except Exception:
        # força build em casos raros
        in_shapes_tmp = [tuple(x.shape) for x in getattr(model, "inputs", [])]
        if in_shapes_tmp:
            shape = list(in_shapes_tmp[0]); shape[0] = 1
            _ = model(tf.random.uniform(shape, dtype=tf.float32), training=False)
        trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
        non_trainable_params = int(np.sum([np.prod(v.shape) for v in model.non_trainable_variables]))
        total_params = int(model.count_params())

    in_shapes, out_shapes, n_classes = _keras_io_shapes(model)

    info = {
        "tipo": "Keras",
        "fonte": src,
        "tamanho": fsize,
        "precisao_numerica": policy,
        "camadas": int(num_layers),
        "parametros_totais": total_params,
        "parametros_treinaveis": trainable_params,
        "parametros_nao_treinaveis": non_trainable_params,
        "neurônios_dense": int(dense_units),
        "dtypes_variaveis": var_dtypes,
        "input_shapes": in_shapes,
        "output_shapes": out_shapes,
        "n_classes": n_classes,
    }
    if details:
        info["hist_camadas"] = _keras_layer_histogram(model)
    return info

def _keras_bench(model: tf.keras.Model, runs=50, warmup=10) -> float:
    # usa primeiro input; substitui None pelo batch=1
    ishapes = [tuple(x.shape) for x in getattr(model, "inputs", [])]
    if not ishapes:
        return float("nan")
    shape = list(ishapes[0])
    shape[0] = 1
    x = tf.random.uniform(shape, dtype=tf.float32)
    # warmup
    for _ in range(warmup):
        _ = model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x, training=False)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / runs  # ms

# ------------------------------ TFLITE ----------------------------------
def _guess_tflite_precision(interpreter: tf.lite.Interpreter) -> str:
    dtypes = {d["dtype"] for d in interpreter.get_tensor_details()}
    if np.float16 in dtypes: return "float16"
    if any(dt in dtypes for dt in (np.int8, np.uint8, np.int16)): return "int8/uint8/int16 (quantizado)"
    return "float32"

def _tflite_io_details(interpreter: tf.lite.Interpreter):
    def qinfo(d):
        qp = d.get("quantization_parameters", {})
        s, z = d.get("quantization", (0.0, 0))
        scales = qp.get("scales", None)
        zpts = qp.get("zero_points", None)
        per_channel = bool(scales is not None and len(np.array(scales).shape) == 1 and len(scales) > 1)
        return {"dtype": str(d["dtype"]),
                "shape": d["shape"].tolist(),
                "scale": float(s),
                "zero_point": int(z),
                "per_channel": per_channel,
                "n_scales": int(len(scales)) if scales is not None else 0}
    ins = [qinfo(d) for d in interpreter.get_input_details()]
    outs = [qinfo(d) for d in interpreter.get_output_details()]
    return ins, outs

def _tflite_op_histogram(interpreter: tf.lite.Interpreter) -> Dict[str,int]:
    try:
        ops = interpreter._get_ops_details()  # API privada, pode não existir
    except Exception:
        return {}
    return dict(Counter([op["op_name"] for op in ops]).most_common())

def _tflite_input_range(inp: Dict[str,Any]) -> str:
    dt = inp["dtype"]
    if "int8" in dt:
        mn, mx = -128, 127
    elif "uint8" in dt:
        mn, mx = 0, 255
    else:
        return "float (sem quantização fixa)"
    scale, zp = inp["scale"], inp["zero_point"]
    real_min = (mn - zp) * scale
    real_max = (mx - zp) * scale
    return f"[{real_min:.4f}, {real_max:.4f}] (escala={scale:.6g}, zp={zp})"

def _analyze_tflite(p: PathLike, details: bool) -> dict:
    p = Path(p)
    interp = tf.lite.Interpreter(model_path=str(p))
    interp.allocate_tensors()

    precision = _guess_tflite_precision(interp)
    size = _file_size(p)
    op_hist = _tflite_op_histogram(interp) if details else {}
    try:
        ops_count = sum(op_hist.values()) if op_hist else len(interp._get_ops_details())
    except Exception:
        ops_count = None

    inputs, outputs = _tflite_io_details(interp)
    input_range = _tflite_input_range(inputs[0]) if inputs else "N/A"

    # “neurônios” = soma de outputs de FULLY_CONNECTED (estimativa)
    dense_units = 0
    if op_hist:
        tdetails = {d["index"]: d for d in interp.get_tensor_details()}
        for op in interp._get_ops_details():
            if "FULLY_CONNECTED" in op["op_name"].upper():
                out_idx = op["outputs"][0]
                shape = tdetails[out_idx].get("shape", None)
                if shape is not None and len(shape) > 0:
                    dense_units += int(shape[-1])

    info = {
        "tipo": "TFLite",
        "fonte": str(p),
        "tamanho": size,
        "precisao_numerica": precision,
        "camadas_ou_operadores": (ops_count if ops_count is not None else "N/A"),
        "neurônios_dense": int(dense_units),
        "entradas": inputs,
        "saidas": outputs,
        "faixa_real_entrada": input_range,
    }
    if details:
        info["hist_operadores"] = op_hist
    return info

def _tflite_bench(interpreter: tf.lite.Interpreter, runs=50, warmup=10) -> float:
    inp = interpreter.get_input_details()[0]
    shape = inp["shape"].tolist()
    shape[0] = 1
    x = np.zeros(shape, dtype=inp["dtype"])
    # warmup
    for _ in range(warmup):
        interpreter.set_tensor(inp["index"], x); interpreter.invoke()
    t0 = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(inp["index"], x); interpreter.invoke()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / runs  # ms

# ------------------------------ API ------------------------------------
def summary(
    obj: Union[KerasLike, PathLike],
    *,
    details: bool = True,
    benchmark: bool = False,
    runs: int = 50,
    warmup: int = 10,
    return_dict: bool = False,
):
    """
    summary("modelos/best_model.keras", details=True, benchmark=True)
    summary("modelos/model_dynamic.tflite", details=True, benchmark=True)
    summary(tf.keras.models.load_model(...))
    """
    if isinstance(obj, tf.keras.Model):
        info = _analyze_keras(obj, details)
        model = obj
        bench_ms = _keras_bench(model, runs=runs, warmup=warmup) if benchmark else None
    else:
        path = Path(obj)
        ext = path.suffix.lower()
        if ext in {".keras", ".h5"} or (path.is_dir() and (path / "saved_model.pb").exists()):
            info = _analyze_keras(path, details)
            if benchmark:
                model = _safe_load_model(path)
                bench_ms = _keras_bench(model, runs=runs, warmup=warmup)
            else:
                bench_ms = None
        elif ext == ".tflite":
            info = _analyze_tflite(path, details)
            if benchmark:
                interp = tf.lite.Interpreter(model_path=str(path)); interp.allocate_tensors()
                bench_ms = _tflite_bench(interp, runs=runs, warmup=warmup)
            else:
                bench_ms = None
        else:
            raise ValueError(f"Não sei inspecionar: {path}. Use .keras/.h5/SavedModel ou .tflite")

    print("\n===== RESUMO DO MODELO =====")
    for k, v in info.items():
        print(f"{k}: {v}")
    if benchmark and bench_ms is not None:
        print(f"latencia_media_batch1: {bench_ms:.3f} ms  (runs={runs}, warmup={warmup})")

    if return_dict:
        return info
