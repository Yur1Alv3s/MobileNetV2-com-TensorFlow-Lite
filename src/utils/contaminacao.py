import re
import numpy as np
import tensorflow as tf

def extract_frame_number(path: str) -> int:
    match = re.search(r"\((\d+)\)", path)
    return int(match.group(1)) if match else -1

def decode_tensor_path(path_tensor):
    path_val = path_tensor.numpy()
    if isinstance(path_val, np.ndarray):
        path_val = path_val[0]
    if isinstance(path_val, bytes):
        return path_val.decode("utf-8")
    return str(path_val)

def extract_frame_ids(dataset: tf.data.Dataset) -> set:
    frames = []
    for path, _ in dataset.unbatch():
        path_str = decode_tensor_path(path)
        frame = extract_frame_number(path_str)
        if frame != -1:
            frames.append(frame)
    return set(frames)

def verificar_contaminacao(train_ds, val_ds, test_ds):
    print("/////////////////////////////////////////////////")
    print("      VERIFICAÇÃO DE CONTAMINAÇÃO ENTRE SETS     ")
    print("/////////////////////////////////////////////////")

    train_frames = extract_frame_ids(train_ds)
    val_frames = extract_frame_ids(val_ds)
    test_frames = extract_frame_ids(test_ds)

    intersect_train_val = train_frames & val_frames
    intersect_train_test = train_frames & test_frames
    intersect_val_test = val_frames & test_frames

    print(f"Train/Val interseção:  {len(intersect_train_val)}")
    print(f"Train/Test interseção: {len(intersect_train_test)}")
    print(f"Val/Test interseção:   {len(intersect_val_test)}")

    if intersect_train_val:
        print("Exemplos em Train/Val:", sorted(list(intersect_train_val))[:10])
    if intersect_train_test:
        print("Exemplos em Train/Test:", sorted(list(intersect_train_test))[:10])
    if intersect_val_test:
        print("Exemplos em Val/Test:", sorted(list(intersect_val_test))[:10])
