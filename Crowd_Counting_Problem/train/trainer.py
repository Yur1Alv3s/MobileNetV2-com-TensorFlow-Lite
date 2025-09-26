# train/trainer.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import tensorflow as tf

from data.datasets import make_dataset, iter_ids
from builder.mnv2_builder import build_mnv2_crowd_s8
from utils.paths import ARTIFACTS, ensure_artifact_dirs, load_cfg
from utils.seed import set_global_seed, seed_for_dataset  # usamos determinismo e seed para dados


# ------------------------- MÉTRICAS & UTILS -------------------------

def mae_count(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    mae_count(y_true: Tensor, y_pred: Tensor) -> Tensor

    Calcula o **MAE de contagem** por batch.
    Entrada:
      - y_true: mapa GT [B, Hs, Ws, 1] (float32).
      - y_pred: mapa previsto [B, Hs, Ws, 1] (float32).
    Saída:
      - Escalar float32: média de |sum(y_true) - sum(y_pred)| no batch.
    """
    true_count = tf.reduce_sum(y_true, axis=[1, 2, 3])   # [B]
    pred_count = tf.reduce_sum(y_pred, axis=[1, 2, 3])   # [B]
    return tf.reduce_mean(tf.abs(true_count - pred_count))


def _history_to_csv(history: tf.keras.callbacks.History, out_csv: Path) -> None:
    """
    _history_to_csv(history: History, out_csv: Path) -> None

    Salva o histórico de treino (por época) em CSV.
    Entrada:
      - history: retorno de model.fit(), com history.history (dict).
      - out_csv: caminho do CSV (ex.: artifacts/metrics/train_history.csv).
    Saída:
      - None (efeito: escreve o arquivo).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hist = history.history
    prefer = ["loss", "val_loss", "mae_count", "val_mae_count"]
    keys = prefer + [k for k in hist.keys() if k not in prefer]
    with open(out_csv, "w") as f:
        f.write(",".join(["epoch"] + keys) + "\n")
        for i in range(len(next(iter(hist.values())))):
            row = [str(i + 1)] + [str(hist.get(k, [""] * (i + 1))[i]) for k in keys]
            f.write(",".join(row) + "\n")


# ---------------------------- TREINADOR -----------------------------

def train_model(cfg: dict) -> Dict[str, float | str]:
    """
    train_model(cfg: dict) -> dict

    Treina o modelo de crowd counting (MobileNetV2 + cabeça de densidade),
    com foco em **reprodutibilidade** de resultados entre execuções.

    Entrada:
      - cfg: dict do YAML. Campos usados (com defaults):
          experiment_name: str = "crowd_mnv2_s8"
          input_size: [W, H] = [512, 512]
          output_stride: int = 8
          batch_size: int = 4
          epochs: int = 100
          lr: float = 1e-4
          early_stopping_patience: int = 8
          augmentations: dict = {"flip": True}
          seed: int = 42
          deterministic: bool = True   # liga caminho determinístico no TF quando suportado
          data_shuffle_seed: int|None = None  # se None, usa a seed global

    Saída:
      - dict resumo:
        {
          "best_model_path": "artifacts/models/<exp>.keras",
          "best_val_mae":  float,
          "best_val_loss": float
        }

    Efeitos colaterais:
      - artifacts/models/<experiment_name>.keras   (melhor checkpoint)
      - artifacts/metrics/train_history.csv        (histórico por época)
      - artifacts/logs/<experiment_name>/          (logs TensorBoard)
    """
    ensure_artifact_dirs()

    # ---------- 1) Determinismo e seeds ----------
    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)

    # (opcional, ajuda no determinismo do pipeline; pode reduzir performance)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # Força o tf.data a ser determinístico (ordem de elementos estável)
    data_opts = tf.data.Options()
    data_opts.experimental_deterministic = True

    # Seed específica para o shuffle do dataset (ordem reprodutível entre runs)
    data_seed = seed_for_dataset(cfg.get("data_shuffle_seed"))

    exp_name = cfg.get("experiment_name", "crowd_mnv2_s8")
    in_w, in_h = cfg.get("input_size", [512, 512])
    batch_size = int(cfg.get("batch_size", 4))
    epochs = int(cfg.get("epochs", 100))
    lr = float(cfg.get("lr", 1e-4))
    patience = int(cfg.get("early_stopping_patience", 8))

    # ---------- 2) Datasets ----------
    # Observação importante:
    # O make_dataset atual faz o shuffle internamente sem seed.
    # Para reprodutibilidade TOTAL da ordem de exemplos por época,
    # recomendamos ajustar data/datasets.py para usar seed=seed_for_dataset(...)
    # dentro do .shuffle(...). Enquanto isso, aplicamos .with_options(deterministic).
    train_ds = make_dataset("train", cfg).with_options(data_opts)
    val_ds   = make_dataset("val",   cfg).with_options(data_opts)

    # (Opcional) Se você quiser forçar aqui um shuffle reprodutível em cima do dataset
    # (caso não queira editar data/datasets.py), descomente o bloco abaixo.
    # Atenção: isso adiciona um segundo shuffle. Ideal é editar o datasets.py.
    #
    # train_ds = train_ds.shuffle(
    #     buffer_size= min(1000, len(iter_ids("train"))),
    #     seed=data_seed,
    #     reshuffle_each_iteration=True
    # )

    n_train = len(iter_ids("train"))
    n_val   = len(iter_ids("val"))
    steps_per_epoch   = math.ceil(n_train / batch_size) if n_train else None
    validation_steps  = math.ceil(n_val / batch_size) if n_val else None

    # ---------- 3) Modelo ----------
    model = build_mnv2_crowd_s8(cfg)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.MeanSquaredError()  # MSE no mapa
    model.compile(optimizer=optimizer, loss=loss, metrics=[mae_count])

    # ---------- 4) Callbacks ----------
    best_model_path = ARTIFACTS / "models" / f"{exp_name}.keras"
    log_dir = ARTIFACTS / "logs" / exp_name
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_mae_count",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=2,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae_count",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=2,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae_count",
            mode="min",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-6,
            verbose=2,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0),
    ]

    # ---------- 5) Treino ----------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------- 6) Histórico ----------
    _history_to_csv(history, ARTIFACTS / "metrics" / "train_history.csv")

    hist = history.history
    best_val_mae  = float(min(hist.get("val_mae_count", [float("inf")])))
    best_val_loss = float(min(hist.get("val_loss",     [float("inf")])))

    return {
        "best_model_path": str(best_model_path),
        "best_val_mae": best_val_mae,
        "best_val_loss": best_val_loss,
    }


# ------------------------- Execução direta --------------------------

if __name__ == "__main__":
    """
    Execução direta:
      $ python -m train.trainer
    """
    cfg = load_cfg()
    summary = train_model(cfg)
    print("\n[trainer] Resumo:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")
