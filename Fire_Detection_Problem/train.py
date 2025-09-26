
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import os
import warnings
import tensorflow as tf
import Fire_Detection_Problem.models.mobileNetv2 as mobilenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preproc # type: ignore
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from Fire_Detection_Problem.utils.contaminacao import verificar_contaminacao
from Fire_Detection_Problem.utils.metrics import *
from Fire_Detection_Problem.utils.logs import *
from Fire_Detection_Problem.converter.tfliteConverter import TFLiteConverter
from Fire_Detection_Problem.loaders import loader


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#############################################################################################################################

# train_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Fire_Detection_Problem/data/Flame2/train")
# val_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Fire_Detection_Problem/data/Flame2/val")
# test_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Flame2_Fire_Detection/data/Flame2/test")


# ds_treino = loader.load_dataset_aug(train_dir, batch_size=32, img_size=(224,224), use_augmentation=True, preprocess_fn=mb_preproc, shuffle=True)
# ds_val   = loader.load_dataset_aug(val_dir,   batch_size=32, img_size=(224,224), use_augmentation=False, preprocess_fn=mb_preproc, shuffle=False)


# monitorDeAprendizado = EarlyStopping(
#     monitor='val_loss',      # ou 'val_accuracy', mas o loss é melhor para evitar overfitting
#     patience=3,              # vai espera 3 épocas sem melhora antes de parar
#     restore_best_weights=True
# )

# logIniciandoTreino()

# modelo = mobilenet.build_model(input_shape=(224, 224, 3),fine_tune=False, unfreeze_last_n=31)


# verificar_contaminacao(ds_treino, ds_val, ds_teste)

# # Chamar para cada conjunto
# contar_labels(ds_treino, "treino")
# contar_labels(ds_val, "validação")
# contar_labels(ds_test, "teste")


# modelo.fit(
#     ds_treino, 
#     validation_data=ds_val,
#     epochs=15,
#     callbacks=[monitorDeAprendizado]
# )

# modelo.save("Modelos/mobilenetV2_Flame2_FineTuning_TrainAugmentation_preProMBNV2_ShuffleTrainONvalOFF_15_Epochs.keras")

# ===================================================================================
# Conversão TF-Lite
# ===================================================================================

# Caminho do modelo treinado 
caminho_keras = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos/mobilenetV2_Flame2_FineTuning_sem_augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs.keras") 

# Criar conversor 
conversor = TFLiteConverter(caminho_keras, nome_saida="TFLITE_mobilenetV2_Flame2_FineTuning_sem_augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs") 

# Converter para TFLite 
arquivo_tflite = conversor.converter_tflite(Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Modelos"), quantizacao=None)