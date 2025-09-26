# convert/tf_lite_converter.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Generator, Iterable, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def _load_model(m_or_path: Union[str, Path, tf.keras.Model]) -> Tuple[tf.keras.Model, Optional[Path]]:
    """
    Carrega um modelo Keras a partir de um caminho (.keras/.h5/SavedModel) ou usa o objeto já carregado.
    Retorna (modelo, caminho_keras_se_existir).
    """
    if isinstance(m_or_path, tf.keras.Model):
        return m_or_path, None
    p = Path(m_or_path)
    if not p.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {p}")
    # compile=False para não exigir losses/metrics
    model = tf.keras.models.load_model(str(p), compile=False)
    return model, p


def _infer_input_signature(model: tf.keras.Model) -> Tuple[int, int, int]:
    """
    Deduz (H, W, C) da primeira entrada do modelo.
    Suporta formatos (None,H,W,C) ou (H,W,C).
    """
    ish = model.input_shape
    if isinstance(ish, (list, tuple)) and isinstance(ish[0], (list, tuple)):
        ish = ish[0]
    if len(ish) == 4:
        _, H, W, C = ish
    elif len(ish) == 3:
        H, W, C = ish
    else:
        raise ValueError(f"Formato de input não suportado: {ish}")
    H = int(H); W = int(W); C = int(C)
    if H <= 0 or W <= 0 or C <= 0:
        raise ValueError(f"Tamanhos de input inválidos: {ish}")
    return H, W, C


def _size_mb(p: Union[str, Path]) -> float:
    p = Path(p)
    return os.path.getsize(p) / (1024 * 1024)


def _print_size_compare(maybe_keras_path: Optional[Path], tflite_path: Union[str, Path]) -> None:
    print("\n========== COMPARAÇÃO DE TAMANHO ==========")
    try:
        if maybe_keras_path and maybe_keras_path.exists():
            print(f"Modelo original (.keras): { _size_mb(maybe_keras_path):.2f} MB")
        else:
            print("Modelo original (.keras): (não disponível)")
        print(f"Modelo convertido (.tflite): {_size_mb(tflite_path):.2f} MB")
    except Exception as e:
        print(f"[WARN] Falha ao calcular tamanhos: {e}")
    print("==========================================\n")


# ------------------------------------------------------------
# Representative dataset (opcional, para INT8)
# ------------------------------------------------------------

def _rep_dataset_from_zeros(model: tf.keras.Model, n: int = 128) -> Generator[Iterable[np.ndarray], None, None]:
    """
    Gera n amostras de zeros no formato do input do modelo.
    ÚTIL APENAS para permitir a conversão INT8 rodar; calibra mal.
    Preferir um gerador baseado no seu dataset real.
    """
    H, W, C = _infer_input_signature(model)
    sample = np.zeros((1, H, W, C), dtype=np.float32)
    for _ in range(n):
        yield [sample]


# ------------------------------------------------------------
# Conversão "igual ao binário" (paridade justa para o TCC)
# ------------------------------------------------------------

def convert_like_classifier(
    keras_model_or_path: Union[str, Path, tf.keras.Model],
    out_dir: Union[str, Path],
    nome_saida: str = "crowd_likebin",
    quantizacao: bool = True,
) -> str:
    """
    Converte o modelo de CONTAGEM com as MESMAS configurações do seu conversor de
    CLASSIFICAÇÃO BINÁRIA (tfliteConverter.py):

      - converter = TFLiteConverter.from_keras_model(model)
      - se quantizacao=True -> converter.optimizations = [tf.lite.Optimize.DEFAULT]
      - NÃO define supported_types (sem FP16)
      - NÃO usa representative_dataset (sem INT8 full-integer)
      - Mantém I/O em float32

    Retorna o caminho do .tflite salvo.
    """
    model, keras_path = _load_model(keras_model_or_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{nome_saida}.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantizacao:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)

    _print_size_compare(keras_path, out_path)
    print(f"[INFO] Modelo TFLite salvo em: {out_path}")
    return str(out_path)


# ------------------------------------------------------------
# Caminhos adicionais (opcionais): FP16 e INT8
# ------------------------------------------------------------

def convert_to_tflite(
    keras_model_or_path: Union[str, Path, tf.keras.Model],
    out_dir: Union[str, Path],
    nome_saida: str = "crowd_model",
    mode: str = "fp16",  # "fp16" | "int8"
    representative_fn: Optional[Callable[[], Generator[Iterable[np.ndarray], None, None]]] = None,
    force_float_io_for_int8: bool = False,
) -> str:
    """
    Conversor flexível:
      - FP16: usa Optimize.DEFAULT + target_spec.supported_types = [tf.float16]
      - INT8: usa Optimize.DEFAULT + representative_dataset (se fornecido)

    Args
    ----
    keras_model_or_path: caminho ou objeto Keras.
    out_dir: pasta de saída.
    nome_saida: nome do arquivo .tflite (sem extensão).
    mode: "fp16" ou "int8".
    representative_fn: gerador de amostras (obrigatório para INT8 decente).
    force_float_io_for_int8: se True, mantém I/O em float32 (útil para compatibilidade).

    Retorna: caminho do .tflite salvo.
    """
    model, keras_path = _load_model(keras_model_or_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{nome_saida}.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    mode = mode.lower().strip()

    if mode == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        # Mantém I/O em float32 por padrão (boa compatibilidade)
    elif mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Representative dataset
        if representative_fn is None:
            # fallback: zeros (funciona, calibra mal)
            representative_fn = lambda: _rep_dataset_from_zeros(model, n=128)
        converter.representative_dataset = representative_fn

        # Ops INT8 (full integer) + I/O
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if force_float_io_for_int8:
            # Mantém I/O em float32 (útil p/ facilitar integração)
            pass
        else:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    else:
        raise ValueError("mode deve ser 'fp16' ou 'int8'.")

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)

    _print_size_compare(keras_path, out_path)
    print(f"[INFO] Modelo TFLite salvo em: {out_path}")
    return str(out_path)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="Conversor TFLite para modelo de contagem de pessoas."
    )
    p.add_argument("--keras", type=str, required=True, help="Caminho para .keras/.h5/SavedModel")
    p.add_argument("--outdir", type=str, default="artifacts/models", help="Diretório de saída")
    p.add_argument("--name", type=str, default=None, help="Nome base do .tflite (sem extensão)")
    p.add_argument(
        "--mode",
        type=str,
        default="like_classifier",
        choices=["like_classifier", "drq", "fp16", "int8"],
        help=(
            "like_classifier/drq: apenas Optimize.DEFAULT (paridade com binário, I/O float32). "
            "fp16: meia-precisão. int8: requer representative dataset (usa zeros por fallback)."
        ),
    )
    p.add_argument("--no-quant", action="store_true",
                   help="Usado com like_classifier/drq: desativa Optimize.DEFAULT (gera FP32).")
    p.add_argument("--int8-float-io", action="store_true",
                   help="INT8 com I/O em float32 (compatibilidade).")
    return p


def main():
    args = _build_argparser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    name = args.name
    if name is None:
        base = Path(args.keras).stem
        if args.mode == "like_classifier" or args.mode == "drq":
            name = f"{base}_likebin"
        else:
            name = f"{base}_{args.mode}"

    if args.mode in ("like_classifier", "drq"):
        # Paridade com o conversor binário (tfliteConverter.py)
        quant = not args.no_quant  # por padrão, aplica Optimize.DEFAULT
        path = convert_like_classifier(
            keras_model_or_path=args.keras,
            out_dir=outdir,
            nome_saida=name,
            quantizacao=quant,
        )
    elif args.mode == "fp16":
        path = convert_to_tflite(
            keras_model_or_path=args.keras,
            out_dir=outdir,
            nome_saida=name,
            mode="fp16",
        )
    elif args.mode == "int8":
        path = convert_to_tflite(
            keras_model_or_path=args.keras,
            out_dir=outdir,
            nome_saida=name,
            mode="int8",
            representative_fn=None,            # fallback: zeros internos
            force_float_io_for_int8=args.int8_float_io,
        )
    else:
        raise RuntimeError("Modo desconhecido.")

    print(f"[OK] TFLite salvo em: {path}")


if __name__ == "__main__":
    main()
