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
def representative_dataset_generator(base_dir: Path,
                                     img_size=(224, 224),
                                     rep_samples: int = 500,
                                     batch_size: int = 1,
                                     use_augmentation: bool = False,
                                     preprocess_fn=None):
    """Gera um generator representativo p/ calibração INT8 (TFLite)."""
    fire_dir = base_dir / 'fire'
    nofire_dir = base_dir / 'nofire'
    if not fire_dir.exists() or not nofire_dir.exists():
        raise ValueError(f"Esperado subpastas 'fire' e 'nofire' em {base_dir}")

    fire_files = tf.data.Dataset.list_files(str(fire_dir / '*'))
    nofire_files = tf.data.Dataset.list_files(str(nofire_dir / '*'))

    def _load_image_from_path(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img.set_shape([None, None, 3])
        img = _resize_pad_and_preprocess(img, img_size, preprocess_fn)
        return img

    fire_ds = fire_files.map(_load_image_from_path, num_parallel_calls=AUTOTUNE).batch(batch_size)
    nofire_ds = nofire_files.map(_load_image_from_path, num_parallel_calls=AUTOTUNE).batch(batch_size)

    if use_augmentation:
        aug = _augmenter()
        fire_ds = fire_ds.map(lambda x: aug(x, training=False))
        nofire_ds = nofire_ds.map(lambda x: aug(x, training=False))

    ds = tf.data.Dataset.zip((fire_ds.repeat(), nofire_ds.repeat()))
    ds = ds.map(lambda f_batch, n_batch: tf.concat([f_batch, n_batch], axis=0), num_parallel_calls=AUTOTUNE)

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

# ====== REGRESSÃO (CSV com filename,value) ======
def load_dataset_regression(
    train_dir: Path,
    val_dir: Path,
    labels_filename: str = 'labels.csv',
    img_size: tuple = (224, 224),
    batch_size: int = 32,
    preprocess_fn: Optional[Callable] = None,
    use_augmentation: bool = False,
    shuffle_buffer: int = 512,
    cache: bool = True,
    prefetch: bool = True,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Carrega datasets de regressão (train, val) com labels contínuos (filename,value)."""

    def _read_csv_labels(dir_path: Path) -> tuple[list[str], list[float]]:
        csv_path = dir_path / labels_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Arquivo de labels não encontrado: {csv_path}")
        filepaths: list[str] = []
        values: list[float] = []
        import csv as _csv
        with open(csv_path, 'r', newline='') as f:
            reader = _csv.reader(f)
            for row_idx, row in enumerate(reader):
                if not row:
                    continue
                if row_idx == 0:
                    try:
                        float(row[1])
                    except Exception:
                        continue  # header
                try:
                    filename, value = row[0], float(row[1])
                except Exception as e:
                    raise ValueError(f"Linha inválida em {csv_path}: {row}") from e
                img_path = dir_path / filename
                if not img_path.exists():
                    raise FileNotFoundError(f"Imagem listada no CSV não encontrada: {img_path}")
                filepaths.append(str(img_path))
                values.append(value)
        if not filepaths:
            raise ValueError(f"Nenhuma entrada válida encontrada em {csv_path}")
        return filepaths, values

    def _build_ds(dir_path: Path, training: bool) -> tf.data.Dataset:
        filepaths, values = _read_csv_labels(dir_path)
        paths_ds = tf.data.Dataset.from_tensor_slices(filepaths)
        values_ds = tf.data.Dataset.from_tensor_slices(values)
        ds = tf.data.Dataset.zip((paths_ds, values_ds))

        def _load(path, y):
            img_bytes = tf.io.read_file(path)
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = _resize_pad_and_preprocess(img, img_size, preprocess_fn)
            return img, tf.cast(y, tf.float32)

        ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

        if training and use_augmentation:
            aug = _augmenter()
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)

        if training and shuffle:
            ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

        ds = ds.batch(batch_size)

        if cache:
            ds = ds.cache()
        if prefetch:
            ds = ds.prefetch(AUTOTUNE)

        return ds

    train_ds = _build_ds(train_dir, training=True)
    val_ds = _build_ds(val_dir, training=False)
    return train_ds, val_ds
