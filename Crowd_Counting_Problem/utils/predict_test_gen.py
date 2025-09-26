# utils/predict_test_gen.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf

from utils.paths import ARTIFACTS, img_dir, load_cfg
from data.datasets import iter_ids


# ----------------------- utilitários comuns -------------------------

def _counts_from_maps(maps: np.ndarray | tf.Tensor) -> np.ndarray:
    """
    _counts_from_maps(maps) -> np.ndarray
    Soma mapas [H,W,1] (ou [B,H,W,1]) em contagens. Saída: [B] (ou [1]).
    """
    if isinstance(maps, tf.Tensor):
        maps = maps.numpy()
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim == 3:
        maps = maps[None, ...]
    return maps.sum(axis=(1, 2, 3))


def _decode_and_preprocess(img_path: tf.Tensor, in_w: int, in_h: int) -> tf.Tensor:
    """
    _decode_and_preprocess(img_path, in_w, in_h) -> tf.Tensor[H,W,3] em [-1,1]
    Lê JPEG, redimensiona e normaliza como no treino.
    """
    data = tf.io.read_file(img_path)
    x = tf.image.decode_jpeg(data, channels=3, dct_method="INTEGER_ACCURATE")
    x = tf.image.resize(x, (in_h, in_w), method=tf.image.ResizeMethod.BILINEAR)
    x = tf.image.convert_image_dtype(x, tf.float32)  # [0,1]
    x = (x * 2.0) - 1.0                              # [-1,1]
    x.set_shape([in_h, in_w, 3])
    return x


def _to_uint8_img(x_minus1_1: np.ndarray) -> np.ndarray:
    """
    _to_uint8_img(x_minus1_1) -> uint8 [H,W,3]
    Converte imagem de [-1,1] para [0,255] uint8 para salvar prévias.
    """
    x = ((x_minus1_1 + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return x


# ---------------------------- KERAS ----------------------------------

def _keras_predict_map(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    """
    _keras_predict_map(model, batch) -> np.ndarray [B,Hs,Ws,1]
    Faz forward pass no Keras e retorna os mapas de densidade (float32).
    """
    y = model(batch, training=False)  # tf.Tensor
    return np.asarray(y.numpy(), dtype=np.float32)


# ---------------------------- TFLITE ---------------------------------

def _load_tflite_interpreter(tflite_path: str | Path, delegate: str = "cpu") -> tf.lite.Interpreter:
    """
    _load_tflite_interpreter(path, delegate) -> Interpreter alocado.
    """
    path = str(tflite_path)
    if delegate.lower() == "gpu":
        try:
            interpreter = tf.lite.Interpreter(
                model_path=path,
                experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')]
            )
        except Exception:
            interpreter = tf.lite.Interpreter(model_path=path)
    elif delegate.lower() == "nnapi":
        try:
            interpreter = tf.lite.Interpreter(
                model_path=path,
                experimental_delegates=[tf.lite.experimental.load_delegate('libnnapi_delegate.so')]
            )
        except Exception:
            interpreter = tf.lite.Interpreter(model_path=path)
    else:
        interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def _tflite_predict_map(interpreter: tf.lite.Interpreter, batch: np.ndarray) -> np.ndarray:
    """
    _tflite_predict_map(interpreter, batch) -> np.ndarray [B,Hs,Ws,1] float32
    Faz forward pass no TFLite e retorna mapas de densidade em float.
    Faz (de)quantização automática se o modelo for INT8.
    """
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    x = batch  # [B,H,W,3] float32 em [-1,1]
    if inp["dtype"] == np.int8:
        scale, zero = inp["quantization"]
        xq = (x / scale + zero).round().clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], xq)
    else:
        interpreter.set_tensor(inp["index"], x.astype(np.float32))

    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])

    if out["dtype"] == np.int8:
        scale, zero = out["quantization"]
        y = (y.astype(np.float32) - float(zero)) * float(scale)

    return y.astype(np.float32)


# ----------------------- pré-visualizações ---------------------------

def _save_preview(img_minus1_1: np.ndarray, den_map: np.ndarray, out_path: Path) -> None:
    """
    _save_preview(img_minus1_1, den_map, out_path) -> None
    Salva uma sobreposição simples do heatmap na imagem.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # upsample do mapa para o tamanho da imagem
    H, W, _ = img_minus1_1.shape
    den = den_map.squeeze()
    den_up = tf.image.resize(den[..., None], (H, W), method="bilinear").numpy().squeeze()

    # normaliza para [0,1] visualmente
    vmax = float(np.percentile(den_up, 99.0)) or 1e-6
    vis = (den_up / vmax).clip(0, 1)

    # cria heatmap em uint8 via matplotlib colormap (sem necessidade de exibir)
    import matplotlib.cm as cm
    cmap = cm.get_cmap("jet")
    heat = (cmap(vis)[:, :, :3] * 255.0).astype(np.uint8)

    base = _to_uint8_img(img_minus1_1)
    # blend simples
    out = (0.6 * base + 0.4 * heat).clip(0, 255).astype(np.uint8)

    # salva com imageio (leve), cai para PIL se não tiver

    from PIL import Image
    Image.fromarray(out).save(out_path)


# ------------------------- função principal --------------------------

def predict_test_gen(
    model_path: str | Path,
    cfg: dict | None = None,
    split: str = "test",
    batch_size: Optional[int] = None,
    save_maps: bool = False,
    previews: int = 0,
    delegate: str = "cpu",
) -> str:
    """
    predict_test_gen(model_path, cfg=None, split='test', batch_size=None,
                     save_maps=False, previews=0, delegate='cpu') -> str

    Gera predições de contagem para um split SEM GT (ex.: 'test').
    Detecta automaticamente se 'model_path' é .keras ou .tflite.

    Entradas:
      - model_path: caminho para .keras OU .tflite.
      - cfg: dicionário de config (usa 'input_size' e 'experiment_name'); se None, chama load_cfg().
      - split: normalmente 'test', mas pode ser outro (desde que exista a lista e imagens).
      - batch_size: sobrescreve cfg['batch_size'] se informado.
      - save_maps: salva mapas preditos em artifacts/preds/<split>/<id>.npy
      - previews: número de pré-visualizações (heatmap sobre imagem) para salvar em artifacts/previews/<split>/
      - delegate: 'cpu' | 'gpu' | 'nnapi' (apenas para .tflite)
    Saída:
      - caminho do CSV gerado: artifacts/metrics/<exp>_<mode>_<split>_pred.csv
        (mode = 'keras' | 'tflite')
    """
    cfg = cfg or load_cfg()
    in_w, in_h = cfg.get("input_size", [512, 512])
    bs = int(batch_size or cfg.get("batch_size", 4))
    exp = cfg.get("experiment_name", "crowd_mnv2_s8")

    ids = iter_ids(split)
    if not ids:
        raise RuntimeError(f"Nenhum ID encontrado em data/lists/{split}.txt")

    # prepara caminhos de saída
    mode = "tflite" if str(model_path).lower().endswith(".tflite") else "keras"
    out_csv = ARTIFACTS / "metrics" / f"{exp}_{mode}_{split}_pred.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    preds_dir = ARTIFACTS / "preds" / split
    prevs_dir = ARTIFACTS / "previews" / split
    if save_maps:
        preds_dir.mkdir(parents=True, exist_ok=True)
    if previews > 0:
        prevs_dir.mkdir(parents=True, exist_ok=True)

    # carrega modelo
    keras_model: Optional[tf.keras.Model] = None
    tflite_interpreter: Optional[tf.lite.Interpreter] = None
    if mode == "keras":
        keras_model = tf.keras.models.load_model(str(model_path), compile=False)
    else:
        tflite_interpreter = _load_tflite_interpreter(str(model_path), delegate=delegate)

    # stream de caminhos/ids em batches
    img_paths = [str(img_dir(split) / f"{i}.jpg") for i in ids]
    id_ds = tf.data.Dataset.from_tensor_slices((ids, img_paths))

    def _mapper(i: tf.Tensor, p: tf.Tensor):
        return i, _decode_and_preprocess(p, in_w, in_h)

    ds = id_ds.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE).batch(bs).prefetch(tf.data.AUTOTUNE)

    # loop de inferência
    written_previews = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "pred_count"])

        for id_batch, x_batch in ds:
            x_np = x_batch.numpy()  # [B,H,W,3] em [-1,1]

            if mode == "keras":
                maps = _keras_predict_map(keras_model, x_np)  # [B,Hs,Ws,1]
            else:
                maps = _tflite_predict_map(tflite_interpreter, x_np)

            counts = _counts_from_maps(maps)  # [B]

            # salva CSV (id, contagem)
            for _id, _c in zip(id_batch.numpy(), counts):
                w.writerow([_id.decode("utf-8"), float(_c)])

            # salva mapas (opcional)
            if save_maps:
                for _id, _m in zip(id_batch.numpy(), maps):
                    np.save(preds_dir / f"{_id.decode('utf-8')}.npy", _m.astype(np.float32))

            # salva prévias (opcional)
            if previews > 0 and written_previews < previews:
                need = previews - written_previews
                take = min(need, x_np.shape[0])
                for k in range(take):
                    pid = id_batch.numpy()[k].decode("utf-8")
                    _save_preview(x_np[k], maps[k], prevs_dir / f"{pid}.jpg")
                    written_previews += 1
                    if written_previews >= previews:
                        break

    return str(out_csv)

