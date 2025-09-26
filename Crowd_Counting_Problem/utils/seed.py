# utils/seed.py
from __future__ import annotations
import os
import random
from time import time
from typing import Optional

import numpy as np

try:
    import tensorflow as tf  # opcional (só se TF estiver instalado)
except Exception:
    tf = None  # type: ignore

_CURRENT_SEED: Optional[int] = None


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    set_global_seed(seed: int, deterministic: bool=False) -> None

    Fixa a semente global de aleatoriedade para Python, NumPy e TensorFlow.
    Opcionalmente ativa caminhos mais determinísticos no TensorFlow/cuDNN.

    Entrada:
      - seed: inteiro para inicializar os geradores pseudo-aleatórios.
      - deterministic: se True, tenta ligar determinismo de ops do TF/cuDNN.
    Saída:
      - None. (Efeito colateral: configura seeds/variáveis de ambiente.)
    """
    # Python e NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow (se disponível)
    if tf is not None:
        # TF 2.9+: utilitário que seta Python/NumPy/TF
        try:
            tf.keras.utils.set_random_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            tf.random.set_seed(seed)

        if deterministic:
            # torna algumas operações determinísticas (quando suportadas)
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
            try:
                # TF 2.12+: API experimental
                tf.config.experimental.enable_op_determinism(True)  # type: ignore[attr-defined]
            except Exception:
                pass

    global _CURRENT_SEED
    _CURRENT_SEED = seed


def get_global_seed(default: int | None = None) -> int | None:
    """
    get_global_seed(default: int|None=None) -> int|None

    Retorna a última seed configurada por set_global_seed(),
    ou 'default' se nenhuma seed foi definida.
    Entrada:
      - default: valor a retornar caso ainda não haja seed definida.
    Saída:
      - int com a seed atual, ou 'default' (ou None).
    """
    return _CURRENT_SEED if _CURRENT_SEED is not None else default


def seed_for_dataset(seed: int | None = None) -> int:
    """
    seed_for_dataset(seed: int|None=None) -> int

    Gera uma seed para usar em operações como Dataset.shuffle(..., seed=...).
    Se 'seed' não for fornecida, usa a seed global; se não houver, usa o tempo.

    Entrada:
      - seed: inteiro opcional para sobrescrever a seed gerada.
    Saída:
      - int com a seed a ser usada em data pipelines.
    """
    if seed is not None:
        return int(seed)
    if _CURRENT_SEED is not None:
        return int(_CURRENT_SEED)
    # fallback: tempo atual (não determinístico)
    return int(time()) & 0xFFFFFFFF
