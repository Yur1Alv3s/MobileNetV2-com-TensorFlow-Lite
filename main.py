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
from Fire_Detection_Problem.converter.tfliteConverter import TFLiteConverter
from Fire_Detection_Problem.loaders import loader
from Fire_Detection_Problem.utils.model_summary import summary  # ambos funcionam
from pathlib import Path
from Fire_Detection_Problem.loaders.loader import representative_dataset_generator
from Fire_Detection_Problem.utils.metrics import evaluate_models, measure_peak_rss_keras, measure_peak_rss_tflite
from Fire_Detection_Problem.loaders import mdcount_data

# ===================================================================================
# Conversão TF-Lite
# ===================================================================================


# # Caminho do modelo treinado 
caminho_keras = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/Classificao_Flame2.keras") 

# # Criar conversor 
conversor = TFLiteConverter(caminho_keras, nome_saida="TFLITE_FP32_Classificao_Flame2") 

# # Converter para TFLite 
arquivo_tflite = conversor.converter_tflite(Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos"),
                                             quantizacao=None)



# ===============================================================
# Teste do representative_dataset_generator
# ===============================================================
# rep_dir = Path('Fire_Detection_Problem/data/Flame2/train')
# gen = representative_dataset_generator(rep_dir, img_size=(224,224), rep_samples=2, batch_size=2)
# g = gen()
# for i, sample in enumerate(g):
#     print('sample', i, 'shape:', sample[0].shape, 'dtype:', sample[0].dtype)
# ===============================================================

ROOT = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Fire_Detection_Problem/data/ShanghaiTech_Crowd_Counting_Dataset")
IMG_DIR = ROOT / "part_A_final" / "test_data" / "images"
GT_DIR  = ROOT / "part_A_final" / "test_data" / "ground_truth"

# lista de imagens (ex.: IMG_1.jpg, IMG_2.jpg, ...)
filenames = sorted([p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])


# teste_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Fire_Detection_Problem/data/Flame3")
# ds_teste_classification = loader.load_dataset_aug(
#     teste_dir, 
#     batch_size=32, 
#     img_size=(224,224), 
#     use_augmentation=False, 
#     preprocess_fn=mb_preproc, 
#     shuffle=False)


# ds_teste_regression = mdcount_data.build_shanghaitech_dataset(
#     images_dir=IMG_DIR,
#     gts_dir=GT_DIR,
#     split_list=filenames,
#     img_size=(512, 512),
#     batch_size=6,
#     shuffle=False,   # embaralha no treino
#     k=3, beta=0.3,  # MDCount
#     down_factor=8
# )


# keras_regression_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/mdcount_mnv2_partA.keras"
# tflite_regression_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/TFLITE_mdcount_mnv2_partA.tflite"
# keras_classification_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/mobilenetV2_Flame2_FineTuning_sem_augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs.keras"
# tflite_classification_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/TFLITE_mobilenetV2_Flame2_FineTuning_sem_augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs.tflite"


# res = evaluate_models(
#     task_type="classificacao",
#     keras_model_path=keras_classification_model_path,
#     tflite_model_path=tflite_classification_model_path,
#     test_data= ds_teste_classification,
#   # input_size=(512, 512, 3), #Regression
#     input_size=(224, 224, 3), #Classification
#     threshold=0.5,
#     latency_warmup=20,
#     latency_runs=100,
#     num_threads=1,
#     limit_samples=None,
#     )


# peak_keras = measure_peak_rss_keras(
#     task_type="classificacao",
#     keras_model_path="/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/mobilenetV2_Flame2_FineTuning_sem_augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs.keras",
#     test_data= data_set_teste
# )
# print(f"Memória RAM máxima (pico) Keras: {peak_keras:.4f} MB")

# peak_tflite = measure_peak_rss_tflite(
#     task_type="classificacao",
#     tflite_model_path="/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/TFLITE_mobilenetV2_Flame2_FineTuning_sem_augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs.tflite",
#     test_data= data_set_teste
# )
# print(f"Memória RAM máxima (pico) TFLite: {peak_tflite:.4f} MB")





# comparar_modelos_binarios_keras_tflite(
#     keras_model_path,
#     tflite_model_path,
#     data_set_teste,
#     warmup=100,
#     num_samples=None,
#     medir_memoria=True,
#     threshold=0.5,
#     exibir=True,
#     plotar_cm=False
# )



