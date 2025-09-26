import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
try:
    # EfficientNetV2B0 está disponível em tf.keras.applications em versões compatíveis
    from tensorflow.keras.applications import EfficientNetV2B0  # type: ignore
    # re-export preprocess_input para conveniência
    try:
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input  # type: ignore
    except Exception:
        preprocess_input = None
except Exception:
    EfficientNetV2B0 = None
    preprocess_input = None


def build_model(input_shape=(224, 224, 3), variant='b0', fine_tune=False, unfreeze_last_n: int = 30, freeze_bn: bool = True):
    """
    Construi um modelo EfficientNetV2-B0 para detecção de fogo.

    Parâmetros:
    - input_shape: tupla, forma da imagem de entrada
    - variant: atualmente apenas 'b0' é suportado
    - fine_tune: se True libera camadas para fine-tuning

    Retorna:
    - modelo compilado pronto para treinamento/avaliação
    """

    if EfficientNetV2B0 is None:
        raise RuntimeError("EfficientNetV2B0 não está disponível na instalação do TensorFlow. Atualize o TF ou instale uma versão compatível.")

    base_model = EfficientNetV2B0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Por padrão mantemos o backbone congelado
    base_model.trainable = False

    if fine_tune:
        base_model.trainable = True
        # Congela as camadas iniciais, libera só as finais (controlado por unfreeze_last_n)
        for layer in base_model.layers[:-unfreeze_last_n]:
            layer.trainable = False

    # Opcional: garantir que todas as BatchNormalization do backbone estejam congeladas.
    # if freeze_bn:
    #     for layer in base_model.layers:
    #         if isinstance(layer, tf.keras.layers.BatchNormalization):
    #             layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
