# evaluation/eval_jhu.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import tensorflow as tf

from data.jhu_test_loader import list_jhu_test

def _infer_input_shape(model: tf.keras.Model | tf.lite.Interpreter) -> Tuple[int, int, int]:
    """Descobre (H, W, C) de entrada dependendo do tipo de modelo (Keras ou TFLite)."""
    
    if isinstance(model, tf.keras.Model):
        # Para Keras, usamos model.input_shape
        ish = model.input_shape
        if isinstance(ish, list):  # modelos com múltiplas entradas
            ish = ish[0]
        # Formatos típicos: (None, H, W, C) ou (H, W, C)
        H = int(ish[1] if len(ish) == 4 else ish[0])
        W = int(ish[2] if len(ish) == 4 else ish[1])
        C = int(ish[3] if len(ish) == 4 else (ish[2] if len(ish) == 3 else 3))
        return H, W, C
    
    elif isinstance(model, tf.lite.Interpreter):
        # Para TFLite, usamos interpreter.get_input_details()
        input_details = model.get_input_details()
        input_shape = input_details[0]['shape']  # Detalhes do tensor de entrada
        if len(input_shape) != 4:
            raise ValueError(f"Entrada TFLite inesperada (esperado [batch, H, W, C]): {input_shape}")
        batch_size, H, W, C = input_shape  # Ignoramos o batch_size (geralmente -1 ou 1)
        return H, W, C

    else:
        raise ValueError(f"Tipo de modelo desconhecido: {type(model)}")

def _load_img(path: str, target_hw: Tuple[int,int], channels: int) -> tf.Tensor:
    """Lê, redimensiona e normaliza para [0,1]. Se seu modelo já faz rescaling,
    isso não costuma atrapalhar (a diferença vira um fator quase unitário)."""
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=channels, expand_animations=False)
    img = tf.image.resize(img, target_hw, method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img  # (H,W,C), float32

def _count_from_pred(y) -> float:
    """Extrai a contagem prevista de diferentes formatos de saída:
    - mapa de densidade: (1, H, W, 1) -> soma total
    - escalar: (1, 1) ou (1,) -> valor direto
    - vetor/mapa achatado: soma geral."""
    arr = np.asarray(y)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0])
    # batch-first
    if arr.ndim == 2 and arr.shape[0] == 1:
        return float(arr[0].sum())
    # (1,H,W,1) ou similar
    return float(arr.sum())

def _metrics(pred: np.ndarray, gt: np.ndarray):
    err = np.abs(pred - gt)
    mae = err.mean()
    rmse = np.sqrt(((pred - gt) ** 2).mean())
    nae = (err / (gt + 1e-6)).mean()
    return mae, rmse, nae

def eval_jhu(model_path: Optional[str] = None,
             model: Optional[tf.keras.Model] = None,
             jhu_root: str = "data/JHU-Test",
             save_csv: bool = True):
    """Avalia seu modelo (.keras treinado no NWPU) nas imagens do JHU-Test.
    Use EITHER model_path OU model."""
    assert model_path or model, "Passe model_path ou um modelo carregado."
    if model is None:
        model = tf.keras.models.load_model(model_path, compile=False)

    H, W, C = _infer_input_shape(model)
    pairs = list_jhu_test(jhu_root)
    assert len(pairs) > 0, f"Nada encontrado em {jhu_root}"

    preds, gts, names = [], [], []
    for img_path, gt_count in pairs:
        img = _load_img(img_path, (H, W), C)
        x = tf.expand_dims(img, 0)  # (1,H,W,C)
        y = model.predict(x, verbose=0)
        c = _count_from_pred(y)
        preds.append(c)
        gts.append(float(gt_count))
        names.append(Path(img_path).name)

    preds = np.array(preds, dtype=np.float64)
    gts = np.array(gts, dtype=np.float64)
    mae, rmse, nae = _metrics(preds, gts)

    qtd_imgs = len(gts)
    # print(f"[JHU-Test] imagens: {qtd_imgs}")
    # print(f"MAE = {mae:.3f} | RMSE = {rmse:.3f} | NAE = {nae:.3f}")

    if save_csv:
        outdir = Path("artifacts/metrics/")
        outdir.mkdir(parents=True, exist_ok=True)
        out_csv = outdir / "jhu_eval_results.csv"
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "gt_count", "pred_count", "abs_err"])
            for n, g, p in zip(names, gts, preds):
                w.writerow([n, f"{g:.6f}", f"{p:.6f}", f"{abs(p-g):.6f}"])
        #print(f"→ CSV salvo em: {out_csv}")

    return {"Qtd_imgs":qtd_imgs, "MAE": mae, "RMSE": rmse, "NAE": nae,
            "preds": preds, "gts": gts, "names": names, "out_csv":out_csv}


# Função para avaliação do modelo TFLite
def eval_jhu_lite(tflite_path: str,
                  jhu_root: str = "data/JHU-Test",
                  save_csv: bool = True):
    """Avalia o modelo .tflite (contagem) nas imagens do JHU-Test."""
    # Carrega o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Recupera informações da entrada
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_idx = input_details[0]['index']
    out_idx = output_details[0]['index']

    # Verifica formato de entrada e saída
    H, W, C = _infer_input_shape(interpreter)
    pairs = list_jhu_test(jhu_root)
    assert len(pairs) > 0, f"Nada encontrado em {jhu_root}"

    preds, gts, names = [], [], []
    for img_path, gt_count in pairs:
        img = _load_img(img_path, (H, W), C)
        x = tf.expand_dims(img, 0)  # (1, H, W, C)

        # Executa a inferência no modelo TFLite
        interpreter.set_tensor(in_idx, x)
        interpreter.invoke()

        # Obtém a previsão
        y = interpreter.get_tensor(out_idx)

        # Extrai contagem da previsão
        c = _count_from_pred(y)
        preds.append(c)
        gts.append(float(gt_count))
        names.append(Path(img_path).name)

    preds = np.array(preds, dtype=np.float64)
    gts = np.array(gts, dtype=np.float64)
    mae, rmse, nae = _metrics(preds, gts)

    qtd_imgs = len(gts)

    if save_csv:
        outdir = Path("artifacts/metrics/")
        outdir.mkdir(parents=True, exist_ok=True)
        out_csv = outdir / "jhu_eval_results_tflite.csv"
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "gt_count", "pred_count", "abs_err"])
            for n, g, p in zip(names, gts, preds):
                w.writerow([n, f"{g:.6f}", f"{p:.6f}", f"{abs(p-g):.6f}"])

    return {"Qtd_imgs": qtd_imgs, "MAE": mae, "RMSE": rmse, "NAE": nae,
            "preds": preds, "gts": gts, "names": names, "out_csv": out_csv}
