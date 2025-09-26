import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_preproc # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preproc # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import csv
import os
from datetime import datetime
from Fire_Detection_Problem.dataset import loader
from Fire_Detection_Problem.utils.model_summary import summary  # ambos funcionam
from pathlib import Path
from Fire_Detection_Problem.dataset.loader import representative_dataset_generator
from Fire_Detection_Problem.utils.metrics import comparar_modelos_binarios_keras_tflite

# ===============================================================
# Teste do representative_dataset_generator
# ===============================================================
# rep_dir = Path('Fire_Detection_Problem/data/Flame2/train')
# gen = representative_dataset_generator(rep_dir, img_size=(224,224), rep_samples=2, batch_size=2)
# g = gen()
# for i, sample in enumerate(g):
#     print('sample', i, 'shape:', sample[0].shape, 'dtype:', sample[0].dtype)
# ===============================================================


# ===============================================================
# Execução Principal
# ===============================================================

teste_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Fire_Detection_Problem/data/Flame3")

data_set_teste = loader.load_dataset_aug(teste_dir, batch_size=32, img_size=(224,224), use_augmentation=False, preprocess_fn=mb_preproc, shuffle=False)

keras_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/mobilenetV2_Flame2_FineTuning_sem_augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs.keras"
tflite_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/TFLITE_mobilenetV2_Flame2_FineTuning_sem_augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs.tflite"


comparar_modelos_binarios_keras_tflite(
    keras_model_path,
    tflite_model_path,
    data_set_teste,
    warmup=100,
    num_samples=None,
    medir_memoria=True,
    threshold=0.5,
    exibir=True,
    plotar_cm=False
)

