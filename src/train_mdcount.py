# train_mdcount.py
import os
from pathlib import Path
import json
import random
import numpy as np
import tensorflow as tf

from loaders.mdcount_data import build_shanghaitechA_dataset
from models.mdcount_mobilenetv2 import build_mdcount_mobilenetv2, compile_mdcount_model

# ======= paths (ajuste para o seu diretório) =======
ROOT = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/ShanghaiTech_Crowd_Counting_Dataset")  # atualizado
PART_A_TRAIN_IMG = ROOT / "part_A_final/train_data/images"
PART_A_TRAIN_GT  = ROOT / "part_A_final/train_data/ground_truth"
PART_A_TEST_IMG  = ROOT / "part_A_final/test_data/images"
PART_A_TEST_GT   = ROOT / "part_A_final/test_data/ground_truth"

# lista de nomes de arquivos (train/test)
def list_jpgs(p: Path):
    return sorted([f.name for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])

train_names = list_jpgs(PART_A_TRAIN_IMG)
test_names  = list_jpgs(PART_A_TEST_IMG)

# ======= hiperparâmetros (MDCount) =======
IMG_SIZE = (512, 512)  # pode usar 448 se memória apertar
BATCH = 6              # paper
K = 3                  # k-NN
BETA = 0.3             # sigma_i = BETA * mean(d_k)
DOWN = 8               # saída H/8 x W/8
LR = 4e-4
WD = 2e-4
EPOCHS = 100

# ======= datasets =======
train_ds = build_shanghaitechA_dataset(
    images_dir=PART_A_TRAIN_IMG,
    gts_dir=PART_A_TRAIN_GT,
    split_list=train_names,
    img_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    k=K, beta=BETA, down_factor=DOWN
)
val_ds = build_shanghaitechA_dataset(
    images_dir=PART_A_TEST_IMG,
    gts_dir=PART_A_TEST_GT,
    split_list=test_names,      # validação em Part A test (p/ monitorar)
    img_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False,
    k=K, beta=BETA, down_factor=DOWN
)

# ======= modelo =======
model = build_mdcount_mobilenetv2(input_shape=IMG_SIZE + (3,), wd=WD, out_stride=DOWN)
model = compile_mdcount_model(model, lr=LR, wd=WD)

# callbacks simples
ckpt_path = "checkpoints/mdcount_mnv2_a_os8.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_mae_count", mode="min"),
    tf.keras.callbacks.EarlyStopping(monitor="val_mae_count", patience=10, restore_best_weights=True, mode="min"),
    tf.keras.callbacks.TerminateOnNaN()
]

# ======= treino =======
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ======= avaliação final (em Part A test) =======
eval_metrics = model.evaluate(val_ds, return_dict=True)
print("Eval:", eval_metrics)

# ======= export =======
model.save("Modelos/mdcount_mnv2_partA.keras")
