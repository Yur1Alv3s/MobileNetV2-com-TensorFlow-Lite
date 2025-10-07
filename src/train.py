
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import os
import warnings
import tensorflow as tf
import src.models.mobileNetv2 as mobilenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preproc # type: ignore
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from src.utils.contaminacao import verificar_contaminacao
from src.utils.metrics import *
from src.utils.logs import *
from src.converter.tfliteConverter import TFLiteConverter
from src.loaders import loader


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


#####################################################################################################
# Treino de modelo classificador binário (fire / nofire) com MobileNetV2
#####################################################################################################

train_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/splits/train")
val_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/splits/val")

ds_treino = loader.load_dataset_aug(train_dir, batch_size=32, img_size=(224,224), use_augmentation=False, preprocess_fn=mb_preproc, shuffle=True)
ds_val   = loader.load_dataset_aug(val_dir,   batch_size=32, img_size=(224,224), use_augmentation=False, preprocess_fn=mb_preproc, shuffle=False)

modelo = mobilenet.build_model(input_shape=(224, 224, 3),fine_tune=True, unfreeze_last_n=31)

modelo.fit(
    ds_treino, 
    validation_data=ds_val,
    epochs=20
)

modelo.save("Modelos/mobilenetV2_Flame2_FineTuning_no_Augmentation_preProMBNV2_ShuffleTrainONvalOFF_20_Epochs.keras")
