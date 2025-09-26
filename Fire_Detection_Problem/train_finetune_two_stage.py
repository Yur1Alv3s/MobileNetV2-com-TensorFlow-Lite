from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from dataset import loader
try:
    from Fire_Detection_Problem.models import efficientnet
except Exception:
    efficientnet = None

from tensorflow.keras.applications import EfficientNetV2B0


def build_two_stage_model(input_shape=(224,224,3)):
    # base pretrained
    if efficientnet is not None:
        base = efficientnet.EfficientNetV2B0(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        base = EfficientNetV2B0(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model, base


if __name__ == '__main__':
    train_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Fire_Detection_Problem/data/Flame2/train")
    val_dir = Path("/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/Fire_Detection_Problem/data/Flame2/val")

    ds_treino = loader.load_dataset_aug(train_dir, batch_size=32, img_size=(224,224), use_augmentation=False)
    ds_val   = loader.load_dataset_aug(val_dir,   batch_size=32, img_size=(224,224), use_augmentation=False)

    model, base = build_two_stage_model()

    # Stage 1: train head only
    model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    print('Stage 1: training head (base frozen)')
    model.fit(ds_treino, validation_data=ds_val, epochs=2)

    # Stage 2: unfreeze last N layers
    N = 40
    base.trainable = True
    # freeze batchnorm
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    # freeze all except last N
    for layer in base.layers[:-N]:
        layer.trainable = False
    for layer in base.layers[-N:]:
        layer.trainable = True

    model.compile(optimizer=optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    print('Stage 2: fine-tuning last', N, 'layers')
    model.fit(ds_treino, validation_data=ds_val, epochs=2)

    out_path = Path('Modelos/fire_detection_effnV2_finetune_two_stage.keras')
    model.save(out_path)
    print('Model saved to', out_path)
