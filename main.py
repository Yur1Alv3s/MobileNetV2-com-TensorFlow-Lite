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
from src.converter.tfliteConverter import TFLiteConverter
from src.loaders import loader
from src.utils.model_summary import summary  # ambos funcionam
from pathlib import Path
from src.loaders.loader import classification_representative_dataset_generator
from src.loaders.mdcount_data import regression_representative_dataset_generator
from src.utils.metrics import evaluate_models, measure_peak_rss_keras, measure_peak_rss_tflite
from src.loaders import mdcount_data

# ===================================================================================
# Conversão TF-Lite
# ===================================================================================

FLAME2_TRAIN_PATH = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/splits/train")
FLAME2_VAL_PATH = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/splits/val")
FLAME3_TEST_PATH = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/Flame3")
SHANGHAI_TECH_CROWD_PART_A_TRAIN_IMG_PATH = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images")
SHANGHAI_TECH_CROWD_PART_A_TRAIN_GT_PATH  = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth")
ROOT = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/ShanghaiTech_Crowd_Counting_Dataset")
IMG_DIR = ROOT / "part_A_final" / "test_data" / "images"
GT_DIR  = ROOT / "part_A_final" / "test_data" / "ground_truth"
# # # Caminho do modelo treinado 
# caminho_keras = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/Regressao_ShanghaiTech_A.keras") 

# # # Criar conversor 
# conversor = TFLiteConverter(caminho_keras, nome_saida="TFLITE_FP32_Regressao_ShanghaiTech_A") 

# # # Converter para TFLite 
# arquivo_tflite = conversor.converter_tflite(Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos"),
#                                               quantizacao=None)

#conversão int8 full com representative dataset
# arquivo_tflite = conversor.converter_tflite(
#     caminho=Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos"),
#     quantizacao="INT8_FULL",
#     representative_data_dir=SHANGHAI_TECH_CROWD_PART_A_TRAIN_IMG_PATH,
#     rep_samples=300,
#     #img_size=(224, 224),
#     img_size=(512, 512),
#     batch_size=1,
#     task="regression",  # "classification" | "regression"
#     preprocess_fn=mb_preproc,  # se o treino foi com normalizacao /255, passe None para calibrar com /255.0 
# )

# ===================================================================================



# lista de imagens (ex.: IMG_1.jpg, IMG_2.jpg, ...)
filenames = sorted([p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])


teste_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/Flame3")
ds_teste_classification = loader.load_dataset_aug(
    teste_dir, 
    batch_size=32, 
    img_size=(224,224), 
    use_augmentation=False, 
    preprocess_fn=mb_preproc, 
    shuffle=False)


ds_teste_regression = mdcount_data.build_shanghaitech_dataset(
    images_dir=IMG_DIR,
    gts_dir=GT_DIR,
    split_list=filenames,
    img_size=(512, 512),
    batch_size=6,
    shuffle=False,   # embaralha no treino
    k=3, beta=0.3,  # MDCount
    down_factor=8
)


keras_regression_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/Regressao_ShanghaiTech_A.keras"
tflite_regression_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/TFLITE_INT8_DR_Regressao_ShanghaiTech_A.tflite"
keras_classification_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/Classificacao_Flame2.keras"
tflite_classification_model_path = "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/TFLITE_INT8_FULL_Classificacao_Flame2.tflite"


res = evaluate_models(
    task_type="regressao",
    keras_model_path=keras_regression_model_path,
    tflite_model_path=tflite_regression_model_path,
    test_data= ds_teste_regression,
    input_size=(512, 512, 3), #Regression
  # input_size=(224, 224, 3), #Classification
    threshold=0.5,
    latency_warmup=20,
    latency_runs=100,
    num_threads=1,
    limit_samples=None,
    )


# peak_keras = measure_peak_rss_keras(
#     task_type="classificacao",
#     keras_model_path="/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/Classificacao_Flame2.keras",
#     test_data= ds_teste_classification
# )
# print(f"Memória RAM máxima (pico) Keras: {peak_keras:.4f} MB")

# peak_tflite = measure_peak_rss_tflite(
#     task_type="regressao",
#     tflite_model_path="/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/TFLITE_INT8_FULL_Regressao_ShanghaiTech_A.tflite",
#     test_data= ds_teste_regression
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



#Avalia se a conversão foi bem sucedida
# import numpy as np, tensorflow as tf

# path = "Modelos/TFLITE_INT8_FULL_Classificacao_Flame2.tflite"
# interp = tf.lite.Interpreter(model_path=path)
# interp.allocate_tensors()

# inp = interp.get_input_details()[0]
# out = interp.get_output_details()[0]

# print("INPUT:", inp["dtype"], "quantization:", inp["quantization"])
# print("OUTPUT:", out["dtype"], "quantization:", out["quantization"])