from pathlib import Path
import tensorflow as tf
from Fire_Detection_Problem.utils.logs import logGerandoDatasets
import csv
from typing import Tuple, Optional, Callable

AUTOTUNE = tf.data.AUTOTUNE

# ===== Helpers: resize com padding + preprocess =====
def _resize_pad_and_preprocess(img, target_size, preprocess_fn=None):
    """Resize com padding para preservar aspecto + preprocess opcional."""
    h, w = target_size
    img = tf.image.resize_with_pad(img, h, w, method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)  # mantém [0..255] para preprocess tipo MobileNetV2
    if preprocess_fn is not None:
        img = preprocess_fn(img)
    else:
        img = img / 255.0
    return img

def _apply_resize_pipeline(ds, img_size, preprocess_fn=None, training=False, augmenter=None):
    """Aplica resize_with_pad + (opcional) augmentation no dataset (image, y)."""
    def _map(img, y):
        img = _resize_pad_and_preprocess(img, img_size, preprocess_fn)
        if training and augmenter is not None:
            img = augmenter(img, training=True)
        return img, y
    return ds.map(_map, num_parallel_calls=AUTOTUNE)

# ====== CLASSIFICAÇÃO (folders) ======
def create_dataset_from_folders(image_dir: Path, img_size=(224,224), batch_size=32,
                                preprocess_fn=None, shuffle=False, use_augmentation=False):
    """Carrega dataset organizado em pastas (ex.: fire / nofire)."""
    # Lê e gera rótulos automaticamente
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,          # primeiro resize (padrão Keras)
        batch_size=batch_size,
        shuffle=shuffle
    )

    # Augmentation (opcional)
    if use_augmentation:
        aug = _augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Normalização / preprocess (após leitura)
    if preprocess_fn is None:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    else:
        # Reaplica preprocess aqui; se quiser preservar aspecto também na classificação,
        # substitua este bloco por _apply_resize_pipeline(...) lendo os arquivos via tf.data list_files.
        ds = ds.map(lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)

    return ds

def load_dataset(base_dir: Path, batch_size=32, img_size=(224,224), preprocess_fn=None):
    """Mantém comportamento antigo (sem augmentation)."""
    return create_dataset_from_folders(base_dir, img_size=img_size, batch_size=batch_size, preprocess_fn=preprocess_fn)

# ====== Augmenter comum ======
def _augmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1)
    ], name="data_augmentation")

def load_dataset_aug(base_dir: Path, batch_size=32, img_size=(224,224), use_augmentation: bool = False,
                     preprocess_fn=None, shuffle: bool = True):
    """Loader com opção de Data Augmentation."""
    return create_dataset_from_folders(base_dir, img_size=img_size, batch_size=batch_size,
                                       preprocess_fn=preprocess_fn, shuffle=shuffle,
                                       use_augmentation=use_augmentation)

# ====== Representante p/ calibração INT8 ======
def classification_representative_dataset_generator(base_dir: Path,
                                     img_size=(224, 224),
                                     rep_samples: int = 500,
                                     batch_size: int = 1,
                                     use_augmentation: bool = False,
                                     preprocess_fn=None):
    """Gera um generator representativo p/ calibração INT8 (TFLite) para classificação binária."""

    fire_dir = base_dir / 'fire'
    nofire_dir = base_dir / 'nofire'
    if not fire_dir.exists() or not nofire_dir.exists():
        raise ValueError(f"Esperado subpastas 'fire' e 'nofire' em {base_dir}")

    # Embaralha a seleção de arquivos (variedade nas amostras)
    fire_files = tf.data.Dataset.list_files(str(fire_dir / '*'), shuffle=True, seed=42)
    nofire_files = tf.data.Dataset.list_files(str(nofire_dir / '*'), shuffle=True, seed=42)

    def _load_image_from_path(path):
        img_bytes = tf.io.read_file(path)
        # Aceita JPEG/PNG/WebP; evita quebra com PNG
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])

        # Alinha com o comportamento do treino (image_dataset_from_directory + image_size),
        # que redimensiona diretamente para (H, W) sem preservar aspecto.
        h, w = img_size
        img = tf.image.resize(img, [h, w], method='bilinear', antialias=True)

        # Mantém pixels em [0..255] (float32) para então aplicar o mesmo preprocess do treino
        img = tf.cast(img, tf.float32)
        if preprocess_fn is not None:
            img = preprocess_fn(img)      # ex.: MobileNetV2.preprocess_input
        else:
            img = img / 255.0       # normalização padrão

        return img

    fire_ds = fire_files.map(_load_image_from_path, num_parallel_calls=AUTOTUNE).batch(batch_size)
    nofire_ds = nofire_files.map(_load_image_from_path, num_parallel_calls=AUTOTUNE).batch(batch_size)

    # Mantém augmentation desabilitada p/ calibração (sem aleatoriedade)
    if use_augmentation:
        aug = _augmenter()
        fire_ds = fire_ds.map(lambda x: aug(x, training=False), num_parallel_calls=AUTOTUNE)
        nofire_ds = nofire_ds.map(lambda x: aug(x, training=False), num_parallel_calls=AUTOTUNE)

    # Intercala fire/nofire
    ds = tf.data.Dataset.zip((fire_ds.repeat(), nofire_ds.repeat()))
    ds = ds.map(lambda f_batch, n_batch: tf.concat([f_batch, n_batch], axis=0),
                num_parallel_calls=AUTOTUNE)

    # Generator no formato esperado pelo TFLite (lista com um tensor [1,H,W,3] por yield)
    def generator():
        count = 0
        for batch in ds:
            b = batch.numpy()
            for i in range(b.shape[0]):
                if count >= rep_samples:
                    return
                img = b[i:i+1].astype('float32')
                yield [img]
                count += 1

    return generator

