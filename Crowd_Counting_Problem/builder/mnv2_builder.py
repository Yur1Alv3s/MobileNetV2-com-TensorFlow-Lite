# builder/mnv2_builder.py
from __future__ import annotations
from typing import Sequence, Optional

import tensorflow as tf
from tensorflow.keras import layers, Model

def _nm(base: Optional[str], suffix: str) -> Optional[str]:
    """
    _nm(base, suffix) -> str|None
    Helper que concatena 'base' + '/' + 'suffix' e SANITIZA removendo '/' (Keras 3 não permite '/').
    Entrada:
      - base: prefixo opcional.
      - suffix: sufixo obrigatório.
    Saída:
      - nome seguro (str) sem '/', ou None se base for None.
    """
    if base is None:
        return None
    return f"{base}_{suffix}".replace("/", "_")

def _conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    dilation_rate: int | tuple[int, int] = 1,
    use_separable: bool = True,
    name: Optional[str] = None,
) -> tf.Tensor:
    """
    _conv_bn_relu(x, filters, kernel_size=3, strides=1, dilation_rate=1, use_separable=True, name=None) -> Tensor

    Bloco auxiliar: Convolução (separable ou padrão) + BatchNorm + ReLU.
    Entrada:
      - x: tensor de entrada [B, H, W, C].
      - filters: nº de filtros de saída (int).
      - kernel_size: tamanho do kernel (int ou (kH,kW)).
      - strides: passo da convolução (int ou (sH,sW)).
      - dilation_rate: dilatação (int ou (dH,dW)).
      - use_separable: se True, usa SeparableConv2D; se False, usa Conv2D.
      - name: prefixo opcional para nomear camadas (será sanitizado).
    Saída:
      - Tensor de saída com mesmo H,W (se stride=1) e 'filters' canais.
    """
    Conv = layers.SeparableConv2D if use_separable else layers.Conv2D
    y = Conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=False,
        name=_nm(name, "conv"),
    )(x)
    y = layers.BatchNormalization(name=_nm(name, "bn"))(y)
    y = layers.ReLU(name=_nm(name, "relu"))(y)
    return y

def _aspp_block(
    x: tf.Tensor,
    filters: int,
    dilation_rates: Sequence[int] = (2, 4, 8),
    use_separable: bool = True,
    name: str = "aspp",
) -> tf.Tensor:
    """
    _aspp_block(x, filters, dilation_rates=(2,4,8), use_separable=True, name='aspp') -> Tensor

    Cabeça estilo ASPP “leve”: ramos paralelos com diferentes dilatações + concat + projeção.
    Entrada:
      - x: tensor de features do backbone (ex.: stride 16).
      - filters: nº de filtros por ramo.
      - dilation_rates: sequência de taxas de dilatação (ex.: [2,4,8]).
      - use_separable: se True, usa SeparableConv2D nos ramos dilatados.
      - name: nome base do bloco (será sanitizado).
    Saída:
      - Tensor concatenado e projetado (mesmo H,W do x, canais ~filters).
    """
    name = name.replace("/", "_") if name else name
    branches = []

    # 1x1 sem dilatação
    b0 = _conv_bn_relu(x, filters, kernel_size=1, use_separable=False, name=_nm(name, "b0_1x1"))
    branches.append(b0)

    # Ramos dilatados 3x3
    for i, r in enumerate(dilation_rates, start=1):
        bi = _conv_bn_relu(
            x,
            filters,
            kernel_size=3,
            dilation_rate=r,
            use_separable=use_separable,
            name=_nm(name, f"b{i}_r{r}"),
        )
        branches.append(bi)

    y = layers.Concatenate(name=_nm(name, "concat"))(branches)
    y = _conv_bn_relu(y, filters, kernel_size=1, use_separable=False, name=_nm(name, "proj"))
    return y

def build_mnv2_crowd_s8(cfg: dict) -> Model:
    """
    build_mnv2_crowd_s8(cfg: dict) -> keras.Model

    Constrói o modelo de crowd counting baseado em MobileNetV2 com saída de mapa de densidade
    em stride 8 (ex.: entrada 512×512 → saída 64×64×1 com ReLU).

    Entrada (cfg):
      - input_size: [W, H] (ex.: [512, 512])  → define input_shape=(H,W,3).
      - output_stride: int (esperado 8)       → cabeça sobe de 16→8.
      - alpha: float (ex.: 1.0)               → largura da MobileNetV2.
      - weights: 'imagenet' | None            → pesos do backbone.
      - train_backbone: bool (default False)  → se False, congela o backbone.
      - head_filters: int (default 128)       → canais da cabeça.
      - dilation_rates: lista (default [2,4,8]) → ASPP leve.
      - use_separable: bool (default True)    → separable convs na cabeça (leve).
      - backbone_layer: str (default 'block_13_expand_relu') → feature de stride 16.

    Saída:
      - keras.Model com:
          inputs: [B, H, W, 3] float32
          outputs: [B, H/8, W/8, 1] com ReLU (densidade ≥ 0).
    """
    in_w, in_h = cfg.get("input_size", [512, 512])
    stride = int(cfg.get("output_stride", 8))
    if stride != 8:
        raise ValueError(f"Este builder assume output_stride=8; recebido: {stride}")

    alpha = float(cfg.get("alpha", 1.0))
    weights = cfg.get("weights", "imagenet")  # 'imagenet' ou None
    train_backbone = bool(cfg.get("train_backbone", False))
    head_filters = int(cfg.get("head_filters", 128))
    dilation_rates = cfg.get("dilation_rates", [2, 4, 8])
    use_separable = bool(cfg.get("use_separable", True))
    feat_layer = cfg.get("backbone_layer", "block_13_expand_relu")  # stride ~16

    input_shape = (in_h, in_w, 3)
    inputs = layers.Input(shape=input_shape, name="input")

    # Backbone MobileNetV2
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        input_shape=input_shape,
        include_top=False,
        alpha=alpha,
        weights=weights,
        pooling=None,
    )
    try:
        feat = backbone.get_layer(feat_layer).output
    except ValueError as e:
        available = [l.name for l in backbone.layers if "expand_relu" in l.name or "project" in l.name]
        raise ValueError(
            f"Camada '{feat_layer}' não encontrada no MobileNetV2. "
            f"Algumas disponíveis: {available[:8]} ..."
        ) from e

    backbone.trainable = train_backbone

    # Cabeça ASPP leve nas features de stride 16
    x = _aspp_block(
        feat,
        filters=head_filters,
        dilation_rates=dilation_rates,
        use_separable=use_separable,
        name="head_aspp",
    )

    # Upsample ×2 (16 -> 8) + refino
    x = layers.UpSampling2D(size=2, interpolation="bilinear", name="head_upsample16to8")(x)
    x = _conv_bn_relu(x, head_filters // 2, kernel_size=3, use_separable=use_separable, name="head_refine")

    # Saída: 1 canal com ReLU (densidade >= 0)
    out = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="relu",
        name="density",
    )(x)

    model = Model(inputs=inputs, outputs=out, name="mnv2_crowd_s8")
    return model
