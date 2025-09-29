# crowd/mdcount_data.py
import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
import tensorflow as tf

# Dependência leve para k-NN (MDCount usa kNN p/ sigma geom.-adaptativo)
from sklearn.neighbors import NearestNeighbors
import scipy.io as sio  # para ler .mat do ShanghaiTech

AUTOTUNE = tf.data.AUTOTUNE


# ---------- util: ler pontos das .mat do ShanghaiTech ----------
def read_points_from_mat(mat_path: Path) -> np.ndarray:
    """
    Lê anotações de cabeças do arquivo .mat.
    Suporta chaves 'image_info' (oficial) e 'annPoints' (alguns mirrors).
    Retorna array (N,2) no formato [x, y].
    """
    data = sio.loadmat(str(mat_path))
    if "image_info" in data:
        # formato: data['image_info'][0,0][0,0][0] -> (N,2)
        pts = data["image_info"][0, 0][0, 0][0]
    elif "annPoints" in data:
        pts = data["annPoints"]
    else:
        raise KeyError(f"Chaves de GT não encontradas em {mat_path.name}")
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Pontos com forma inválida em {mat_path}: {pts.shape}")
    return pts  # (N,2) [x,y]


# ---------- util: mapeamento p/ resize_with_pad ----------
def map_points_resize_with_pad(points_xy: np.ndarray,
                               orig_hw: Tuple[int, int],
                               target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Mapeia coordenadas (x,y) originais para o sistema após resize_with_pad(H,W).
    """
    H, W = orig_hw
    Ht, Wt = target_hw
    scale = min(Ht / H, Wt / W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    pad_top = (Ht - newH) // 2
    pad_left = (Wt - newW) // 2

    pts = points_xy.copy()
    pts[:, 0] = pts[:, 0] * scale + pad_left  # x
    pts[:, 1] = pts[:, 1] * scale + pad_top   # y
    # também clampa para dentro
    pts[:, 0] = np.clip(pts[:, 0], 0, Wt - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, Ht - 1)
    return pts


# ---------- GT: Gaussianas geom.-adaptativas (k-NN) ----------
def geometry_adaptive_sigmas(points_xy: np.ndarray,
                             k: int = 3,
                             beta: float = 0.3,
                             min_sigma: float = 1.0) -> np.ndarray:
    """
    sigma_i = beta * média das distâncias aos k vizinhos mais próximos do ponto i.
    """
    N = len(points_xy)
    if N == 0:
        return np.zeros((0,), dtype=np.float32)
    if N == 1 or k <= 0:
        # fallback (imagem com 1 cabeça): usa sigma fixo moderado
        return np.full((N,), max(min_sigma, 4.0), dtype=np.float32)

    n_neighbors = min(k + 1, N)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(points_xy)
    dists, _ = nbrs.kneighbors(points_xy)  # inclui a distância 0 a si mesmo
    # média das k distâncias não-nulas
    mean_k = dists[:, 1:n_neighbors].mean(axis=1)
    sigmas = beta * mean_k
    sigmas = np.maximum(sigmas, min_sigma)
    return sigmas.astype(np.float32)


def draw_gaussians(points_xy: np.ndarray,
                   sigmas: np.ndarray,
                   hw: Tuple[int, int],
                   truncate: float = 3.0) -> np.ndarray:
    """
    Rasteriza a soma de Gaussianas normalizadas (integral = 1 por pessoa).
    Retorna mapa de densidade em shape (H, W), float32.
    """
    H, W = hw
    D = np.zeros((H, W), dtype=np.float32)
    for (x, y), s in zip(points_xy, sigmas):
        if s <= 0:  # safety
            continue
        r = int(max(1, truncate * s))
        x0, x1 = max(0, int(x) - r), min(W - 1, int(x) + r)
        y0, y1 = max(0, int(y) - r), min(H - 1, int(y) + r)
        if x1 <= x0 or y1 <= y0:
            continue

        xs = np.arange(x0, x1 + 1) - x
        ys = np.arange(y0, y1 + 1) - y
        X, Y = np.meshgrid(xs, ys)
        G = np.exp(-(X**2 + Y**2) / (2 * (s**2))) / (2.0 * np.pi * (s**2))
        D[y0:y1+1, x0:x1+1] += G.astype(np.float32)
    return D


# ---------- downsample por bloco (soma em blocos) ----------
def block_sum_pool(density: np.ndarray, factor: int) -> np.ndarray:
    """
    Soma não sobreposta em blocos factor×factor (“sum pooling”).
    Mantém a soma total (contagem) inalterada.
    """
    if factor == 1:
        return density
    H, W = density.shape
    Hs = (H // factor) * factor
    Ws = (W // factor) * factor
    density = density[:Hs, :Ws]
    d = density.reshape(Hs // factor, factor, Ws // factor, factor).sum(axis=(1, 3))
    return d.astype(np.float32)


# ---------- pipeline numpy: de imagem+mat -> (img_proc, gt_down) ----------
def prepare_image_and_gt(img_bytes,
                         mat_path_in,
                         target_hw_in,
                         k_in,
                         beta_in,
                         down_factor_in) -> Tuple[np.ndarray, np.ndarray]:
    """
    Versão robusta para ser chamada via tf.numpy_function / tf.py_function.
    Aceita tensores/bytes e converte para tipos Python, faz:
      - decode da imagem
      - resize_with_pad
      - mapeia pontos
      - GT geom.-adaptativo
      - sum-pooling (downsample por soma)
    Retorna:
      img_proc: float32 [-1,1], shape (Ht, Wt, 3)
      gt_down:  float32,        shape (Ht/df, Wt/df)
    """
    # ---- helpers para converter entradas vindas do TF ----
    def _to_str(x):
        import numpy as _np
        if isinstance(x, (bytes, _np.bytes_)):
            return x.decode("utf-8")
        if isinstance(x, _np.ndarray):
            # escalar em ndarray (p.ex. np.bytes_ ou np.str_)
            if x.ndim == 0:
                val = x.item()
                if isinstance(val, (bytes, _np.bytes_)):
                    return val.decode("utf-8")
                return str(val)
            # vetor de 1 elemento
            if x.size == 1:
                val = x.reshape(()).item()
                if isinstance(val, (bytes, _np.bytes_)):
                    return val.decode("utf-8")
                return str(val)
            # lista de strings
            return str(x.tolist())
        return str(x)

    def _to_int_pair(x):
        import numpy as _np
        if isinstance(x, _np.ndarray):
            x = x.astype(_np.int64).ravel()
            if x.size >= 2:
                return int(x[0]), int(x[1])
            elif x.size == 1:
                v = int(x[0])
                return v, v
            else:
                raise ValueError("target_hw vazio")
        if isinstance(x, (tuple, list)):
            return int(x[0]), int(x[1])
        return int(x), int(x)

    def _to_int(x):
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return int(_np.asarray(x).reshape(()).item())
        return int(x)

    def _to_float(x):
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return float(_np.asarray(x).reshape(()).item())
        return float(x)

    # ---- converte entradas ----
    mat_path_str = _to_str(mat_path_in)
    Ht, Wt = _to_int_pair(target_hw_in)
    k = _to_int(k_in)
    beta = _to_float(beta_in)
    down_factor = _to_int(down_factor_in)

    # ---- decode da imagem (np.uint8) ----
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False).numpy()
    H, W = img.shape[0], img.shape[1]

    # ---- resize_with_pad + preprocess MobileNetV2 ----
    img_tf = tf.image.resize_with_pad(img, Ht, Wt, method="bilinear", antialias=True)
    img_proc = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(img_tf, tf.float32)
    ).numpy()

    # ---- lê pontos do .mat e mapeia p/ coords redimensionadas ----
    mat_path = Path(mat_path_str)
    pts = read_points_from_mat(mat_path)  # (N,2) [x,y] no sistema original
    pts_resized = map_points_resize_with_pad(pts, (H, W), (Ht, Wt))

    # ---- sigmas geom.-adaptativos + densidade ----
    sigmas = geometry_adaptive_sigmas(pts_resized, k=k, beta=beta)
    dens = draw_gaussians(pts_resized, sigmas, (Ht, Wt))

    # ---- downsample por soma (mantém soma == N) ----
    gt_down = block_sum_pool(dens, down_factor)

    return img_proc.astype(np.float32), gt_down.astype(np.float32)


# ---------- tf.data builder (on-the-fly via py_function) ----------
def build_shanghaitech_dataset(
    images_dir: Path,
    gts_dir: Path,
    split_list: List[str],
    img_size: Tuple[int, int] = (512, 512),
    batch_size: int = 6,
    shuffle: bool = True,
    k: int = 3,
    beta: float = 0.3,
    down_factor: int = 8,
) -> tf.data.Dataset:
    """
    Cria um tf.data.Dataset de (img, gt_down) para ShanghaiTech Part A ou Part B.
    - images_dir: .../part_A/*/images  ou  .../part_B/*/images
    - gts_dir:    .../part_A/*/ground_truth  ou  .../part_B/*/ground_truth
    - split_list: lista com nomes dos arquivos de imagem (ex.: "IMG_1.jpg")
    """
    def _resolve_mat_path(name: str) -> str:
        # Normalmente: "IMG_1.jpg" -> "GT_IMG_1.mat"
        stem = Path(name).stem  # "IMG_1"
        cand = gts_dir / f"GT_{stem}.mat"
        if cand.exists():
            return str(cand)
        # Fallbacks: alguns mirrors variam o sufixo
        # Tenta padrões como: GT_IMG_1_*.mat
        pats = sorted(gts_dir.glob(f"GT_{stem}*.mat"))
        if len(pats) == 1:
            return str(pats[0])
        if len(pats) > 1:
            # escolhe determinístico (primeiro em ordem alfanumérica)
            return str(pats[0])
        raise FileNotFoundError(f"GT .mat não encontrado p/ {name} em {gts_dir}")

    img_paths = [str(images_dir / name) for name in split_list]
    mat_paths = [ _resolve_mat_path(name) for name in split_list ]
    Ht, Wt = img_size

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mat_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths), reshuffle_each_iteration=True)

    def _load_pair(img_path, mat_path):
        img_bytes = tf.io.read_file(img_path)

        # passe argumentos como tensores (compatível com tf.numpy_function)
        hw     = tf.stack([Ht, Wt])            # int32 [2]
        k_t    = tf.cast(k, tf.int32)
        beta_t = tf.cast(beta, tf.float32)
        df_t   = tf.cast(down_factor, tf.int32)

        img_proc, gt_down = tf.numpy_function(
            func=prepare_image_and_gt,        # mesma função já usada no Part A
            inp=[img_bytes, mat_path, hw, k_t, beta_t, df_t],
            Tout=[tf.float32, tf.float32]
        )
        # shapes estáticas
        img_proc.set_shape([Ht, Wt, 3])
        gt_down.set_shape([Ht // down_factor, Wt // down_factor])

        # adiciona canal à GT p/ casar com saída (H',W',1)
        gt_down = tf.expand_dims(gt_down, axis=-1)
        return img_proc, gt_down

    ds = ds.map(_load_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds
