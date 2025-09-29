# models/mdcount_mobilenetv2.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.applications import mobilenet_v2 # type: ignore
from tensorflow.keras.optimizers import Adam


# ----- ASPP leve com depthwise separable (rates 1,2,3) -----
def aspp_sep(x, filters=96, rates=(1, 2, 3), wd=2e-4):
    outs = []
    for r in rates:
        y = layers.SeparableConv2D(
            filters, 3, padding="same", dilation_rate=r, activation="relu",
            depthwise_regularizer=regularizers.l2(wd),
            pointwise_regularizer=regularizers.l2(wd),
            name=f"aspp_sep_r{r}"
        )(x)
        outs.append(y)
    y = layers.Concatenate(name="aspp_concat")(outs)
    y = layers.Conv2D(filters, 1, activation="relu",
                      kernel_regularizer=regularizers.l2(wd),
                      name="aspp_proj")(y)
    return y


def build_mdcount_mobilenetv2(
    input_shape=(512, 512, 3),
    wd=2e-4,                   # weight decay (paper: 2e-4)
    out_stride=8               # saída no H/8 x W/8
):
    inp = layers.Input(shape=input_shape, name="image")

    # Encoder: MobileNetV2, pesos ImageNet
    base = mobilenet_v2.MobileNetV2(include_top=False, weights="imagenet",
                                    input_tensor=inp, input_shape=input_shape)

    # Pega feature stride~16 e sobe para stride~8
    feat16 = base.get_layer("block_13_expand_relu").output   # stride 16
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up16_to_8")(feat16)

    # ASPP leve (rates 1,2,3)
    x = aspp_sep(x, filters=96, rates=(1, 2, 3), wd=wd)

    # Cabeça de densidade (não-negativa)
    dens = layers.SeparableConv2D(
        64, 3, padding="same", activation="relu",
        depthwise_regularizer=regularizers.l2(wd),
        pointwise_regularizer=regularizers.l2(wd),
        name="head_sepconv"
    )(x)
    dens = layers.Conv2D(
        1, 1, padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(wd),
        name="density_map"    # saída (H/8, W/8, 1)
    )(dens)

    model = models.Model(inp, dens, name="mdcount_mnv2_os8")

    return model


# --------- métricas de contagem (contam soma do mapa) ----------
@tf.function
def _sum_map(dmap):
    # soma por amostra (B,H,W,1) -> (B,)
    return tf.reduce_sum(dmap, axis=[1, 2, 3])

class CountMAE(tf.keras.metrics.Metric):
    def __init__(self, name="mae_count", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mae = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true e y_pred são mapas (H',W',1) com SOMA == contagem
        err = tf.abs(_sum_map(y_true) - _sum_map(y_pred))
        self.mae.update_state(err)

    def result(self):
        return self.mae.result()

    def reset_state(self):
        self.mae.reset_state()

class CountRMSE(tf.keras.metrics.Metric):
    def __init__(self, name="rmse_count", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        se = tf.square(_sum_map(y_true) - _sum_map(y_pred))
        self.mse.update_state(se)

    def result(self):
        return tf.sqrt(self.mse.result())

    def reset_state(self):
        self.mse.reset_state()


def compile_mdcount_model(model: tf.keras.Model, lr=4e-4, wd=2e-4):
    """
    Paper: Adam, lr=4e-4, weight decay=2e-4; loss = MSE no mapa.
    """
    opt = Adam(learning_rate=lr, weight_decay=wd)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[CountMAE(), CountRMSE()]
    )
    return model
