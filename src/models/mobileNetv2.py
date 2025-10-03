import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.applications import mobilenet_v2  # type: ignore # <-- necessário para preprocess e modelo
from tensorflow.keras.optimizers import Adam            # <-- usado no compile


def build_model(input_shape=(224, 224, 3), fine_tune=False, unfreeze_last_n: int = 30, freeze_bn: bool = True):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Por padrão mantemos o backbone congelado (compatível com seu fluxo atual)
    base_model.trainable = False

    if fine_tune:
        # Liberar todas as camadas inicialmente e depois congelar as iniciais, mantendo apenas as últimas 'unfreeze_last_n' treináveis
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_last_n]:
            layer.trainable = False

    # Opção para garantir BNs congeladas mesmo quando fine_tune=True
    # if freeze_bn:
    #     for layer in base_model.layers:
    #         if isinstance(layer, tf.keras.layers.BatchNormalization):
    #             layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),  # LR menor em fine-tuning
        #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)  # para treino sem fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model