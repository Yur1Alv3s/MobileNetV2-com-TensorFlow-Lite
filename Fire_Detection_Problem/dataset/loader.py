from pathlib import Path
import tensorflow as tf
from Fire_Detection_Problem.utils.logs import logGerandoDatasets

AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# Loader simples (sem augmentation) — MANTIDO
# ============================================================
def create_dataset_from_folders(image_dir: Path, img_size=(224,224), batch_size=32, preprocess_fn=None, shuffle=False, use_augmentation=False):
    """Carrega dataset organizado em pastas (ex.: fire / nofire).

    Args:
        preprocess_fn: função de preprocessamento aplicada às imagens. Se None, usa /255.0.
    """
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle
    )

    # Aplicar augmentation (se solicitado) antes do preprocess
    if use_augmentation:
        aug = _augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Normalização / preprocess
    if preprocess_fn is None:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)


    return ds

def load_dataset(base_dir: Path, batch_size=32, img_size=(224,224), preprocess_fn=None):
    """Mantém comportamento antigo (sem augmentation).

    Agora aceita `preprocess_fn` para manter consistência com `load_dataset_aug`.
    """
    return create_dataset_from_folders(base_dir, img_size=img_size, batch_size=batch_size, preprocess_fn=preprocess_fn)


# ============================================================
# Novo loader (com ou sem augmentation)
# ============================================================
def _augmenter():
    """Pipeline de Data Augmentation (aplicado só se use_augmentation=True)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1)
    ], name="data_augmentation")

def load_dataset_aug(base_dir: Path, batch_size=32, img_size=(224,224), use_augmentation: bool = False, preprocess_fn=None, shuffle: bool = True):
    """
    Novo método para carregar dataset com opção de Data Augmentation.

    Args:
        base_dir: pasta raiz (com subpastas de classes).
        batch_size: tamanho do batch.
        img_size: tamanho da imagem (h, w).
        use_augmentation: se True, aplica augmentation.
    """

    return create_dataset_from_folders(base_dir, img_size=img_size, batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=shuffle, use_augmentation=use_augmentation)


def representative_dataset_generator(base_dir: Path,
                                     img_size=(224, 224),
                                     rep_samples: int = 500,
                                     batch_size: int = 1,
                                     use_augmentation: bool = False,
                                     preprocess_fn=None):
    """
    Gera um generator representativo para calibração de quantização INT8.

    Args:
        base_dir: Path para a pasta com imagens organizadas em subpastas por classe.
        img_size: tupla (h, w) do tamanho das imagens que o modelo espera.
        rep_samples: número máximo de amostras a gerar.
        batch_size: tamanho do batch lido do disco (o generator irá iterar por amostras individuais).
        use_augmentation: se True, aplica a mesma augmentation definida no loader (não recomendado para calibração, padrão False).

    Retorna:
        Uma função geradora que pode ser atribuída a `converter.representative_dataset`.
    """

    # Assumimos estrutura por classes: base_dir/fire e base_dir/nofire
    fire_dir = base_dir / 'fire'
    nofire_dir = base_dir / 'nofire'

    if not fire_dir.exists() or not nofire_dir.exists():
        raise ValueError(f"Esperado subpastas 'fire' e 'nofire' em {base_dir}")

    # Lista arquivos de cada classe
    fire_pattern = str(fire_dir / '*')
    nofire_pattern = str(nofire_dir / '*')

    fire_files = tf.data.Dataset.list_files(fire_pattern)
    nofire_files = tf.data.Dataset.list_files(nofire_pattern)

    def _load_image_from_path(path):
        img_bytes = tf.io.read_file(path)
        # Tenta decodificar como JPEG (a maioria das imagens do dataset são .jpg)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        # Em alguns casos a forma pode ser dinâmica; garantir shape mínimo para resize
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        # Aplicar preprocess se fornecido, caso contrário normalizar para [0,1]
        if preprocess_fn is None:
            img = img / 255.0
        else:
            # preprocess_fn deve aceitar tensor float (H,W,C) e retornar tensor já normalizado
            img = preprocess_fn(img)
        return img

    fire_ds = fire_files.map(_load_image_from_path, num_parallel_calls=AUTOTUNE)
    nofire_ds = nofire_files.map(_load_image_from_path, num_parallel_calls=AUTOTUNE)

    # Batch para facilitar iteração por amostras
    fire_ds = fire_ds.batch(batch_size)
    nofire_ds = nofire_ds.batch(batch_size)

    # Opcionalmente aplica augmentation (não recomendado, mas suportado)
    if use_augmentation:
        aug = _augmenter()
        fire_ds = fire_ds.map(lambda x: aug(x, training=False))
        nofire_ds = nofire_ds.map(lambda x: aug(x, training=False))

    # Interleave/merge para gerar amostras balanceadas: alterna batches de fire e nofire
    ds = tf.data.Dataset.zip((fire_ds.repeat(), nofire_ds.repeat()))

    # Flatten os pares de batches em um único fluxo de imagens
    def _flatten_batches(f_batch, n_batch):
        # f_batch e n_batch têm shape (B, H, W, C)
        # Concatena ao longo do eixo 0
        return tf.concat([f_batch, n_batch], axis=0)

    ds = ds.map(_flatten_batches, num_parallel_calls=AUTOTUNE)

    if use_augmentation:
        aug = _augmenter()
        ds = ds.map(lambda x: aug(x, training=False))

    # Definir o generator que yield list[array] como esperado pelo TFLiteConverter
    def generator():
        count = 0
        for batch in ds:
            # batch é um tensor de shape (B, H, W, C) após o flatten
            # Itera por cada imagem no batch
            b = batch.numpy()
            for i in range(b.shape[0]):
                if count >= rep_samples:
                    return
                img = b[i:i+1].astype('float32')
                yield [img]
                count += 1

    return generator
