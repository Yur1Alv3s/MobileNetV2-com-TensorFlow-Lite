"""Ferramentas para avaliar baselines e desempenho do modelo de regressão (crowd counting).

Uso rápido (exemplo CLI):

python -m src.evaluation.regression_baselines \
        --train-dir /caminho/para/train_balanced \
        --val-dir /caminho/para/val \
        --model-path Modelos/mobilenetv2_regression_final_log1p.keras \
        --log-space   # se o modelo foi treinado prevendo log1p(count)

Se quiser apenas calcular baselines (sem carregar modelo): remova --model-path.

Também é exposta a função :func:`evaluate_regression` para uso programático dentro do script de treino.

Saídas principais:
- Baselines: MAE zero / média / mediana
- Métricas do modelo: MAE, RMSE, MedAE, NMAE, Poisson floor (média sqrt(y))
- Erro por bins (quantis) e top-K maiores erros absolutos

Estrutura retornada por ``evaluate_regression``:
{
    'baselines': {...},
    'metrics': {...},
    'gains': {...},
    'bins': [ {...}, ... ],
    'topk': [ {...}, ... ],
    'meta': { 'model_path': str|None, 'log_space': bool }
}
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Helpers de leitura de CSV (reaproveita lógica similar ao loader, mas simples)
# ---------------------------------------------------------------------------

def read_labels_csv(dir_path: Path, labels_filename: str = 'labels.csv') -> Tuple[List[str], np.ndarray]:
    csv_path = dir_path / labels_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {csv_path}")
    filepaths: List[str] = []
    values: List[float] = []
    with open(csv_path, 'r', newline='') as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            if row_idx == 0:
                # Tenta detectar header
                try:
                    float(parts[1])
                except Exception:
                    continue
            try:
                filename = parts[0]
                value = float(parts[1])
            except Exception:
                continue
            img_path = dir_path / filename
            if not img_path.exists():
                raise FileNotFoundError(f"Imagem listada não encontrada: {img_path}")
            filepaths.append(str(img_path))
            values.append(value)
    if not filepaths:
        raise ValueError(f"Nenhuma linha válida em {csv_path}")
    return filepaths, np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Baselines simples
# ---------------------------------------------------------------------------

def compute_simple_baselines(val_counts: np.ndarray, train_counts: np.ndarray) -> dict:
    mean_train = train_counts.mean()
    median_train = np.median(train_counts)
    mae_zero = np.mean(np.abs(val_counts - 0))
    mae_mean = np.mean(np.abs(val_counts - mean_train))
    mae_median = np.mean(np.abs(val_counts - median_train))
    return {
        'mae_zero': mae_zero,
        'mae_mean': mae_mean,
        'mae_median': mae_median,
        'mean_train': mean_train,
        'median_train': median_train,
    }


# ---------------------------------------------------------------------------
# Carregar imagens e gerar predições em batches (sem tf.data para simplicidade aqui)
# ---------------------------------------------------------------------------

def load_and_preprocess_image(path: str, img_size=(224,224), preprocess_fn=None) -> np.ndarray:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)
    if preprocess_fn is None:
        img = img / 255.0
    else:
        img = preprocess_fn(img)
    return img.numpy()


def predict_model(model: tf.keras.Model, filepaths: List[str], batch_size: int = 32, img_size=(224,224)) -> np.ndarray:
    preprocess_fn = None
    # Tenta inferir se é MobileNetV2 pelo nome do modelo para aplicar preprocess
    if 'mobilenet' in model.name.lower():
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preproc  # type: ignore
            preprocess_fn = mb_preproc
        except Exception:
            pass
    preds = []
    batch = []
    for fp in filepaths:
        batch.append(load_and_preprocess_image(fp, img_size=img_size, preprocess_fn=preprocess_fn))
        if len(batch) == batch_size:
            batch_arr = np.stack(batch, axis=0)
            p = model.predict(batch_arr, verbose=0)
            preds.append(p.squeeze())
            batch = []
    if batch:
        batch_arr = np.stack(batch, axis=0)
        p = model.predict(batch_arr, verbose=0)
        preds.append(p.squeeze())
    preds_arr = np.concatenate([np.atleast_1d(a) for a in preds], axis=0)
    return preds_arr


# ---------------------------------------------------------------------------
# Avaliação das métricas principais
# ---------------------------------------------------------------------------

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    abs_err = np.abs(y_true - y_pred)
    mae = abs_err.mean()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    medae = np.median(abs_err)
    nmae = mae / (y_true.mean() + 1e-8)
    poisson_floor = np.mean(np.sqrt(y_true))
    return {
        'mae': mae,
        'rmse': rmse,
        'medae': medae,
        'nmae': nmae,
        'poisson_floor': poisson_floor,
    }


# ---------------------------------------------------------------------------
# Erro por bins de quantis
# ---------------------------------------------------------------------------

def error_by_quantile_bins(y_true: np.ndarray, y_pred: np.ndarray, quantiles=(0, 0.5, 0.75, 0.9, 0.975, 1.0)) -> list:
    qs = np.quantile(y_true, quantiles)
    indices = np.digitize(y_true, qs[1:-1], right=True)
    results = []
    for b in range(len(qs) - 1):
        mask = indices == b
        if mask.sum() == 0:
            continue
        bin_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        results.append({
            'bin': b,
            'range': (float(qs[b]), float(qs[b+1])),
            'n': int(mask.sum()),
            'mae': float(bin_mae)
        })
    return results


# ---------------------------------------------------------------------------
# Top-K maiores erros
# ---------------------------------------------------------------------------

def top_k_errors(y_true: np.ndarray, y_pred: np.ndarray, filepaths: List[str], k: int = 10) -> list:
    abs_err = np.abs(y_true - y_pred)
    idx = np.argsort(-abs_err)[:k]
    out = []
    for i in idx:
        out.append({
            'filepath': filepaths[i],
            'y_true': float(y_true[i]),
            'y_pred': float(y_pred[i]),
            'abs_err': float(abs_err[i])
        })
    return out


# ---------------------------------------------------------------------------
# Função programática principal
# ---------------------------------------------------------------------------
def evaluate_regression(
    train_dir: Path,
    val_dir: Path,
    labels_filename: str = 'labels.csv',
    model_path: Optional[Path] = None,
    log_space: bool = False,
    scale_output: float = 1.0,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    top_k: int = 10,
    print_results: bool = False,
) -> Dict[str, Any]:
    """Executa avaliação completa retornando dicionário estruturado.

    Parameters
    ----------
    train_dir : Path
        Diretório de treino (contendo labels.csv).
    val_dir : Path
        Diretório de validação.
    model_path : Path | None
        Caminho para modelo salvo (.keras). Se None retorna apenas baselines.
    log_space : bool
        Indica se a saída do modelo está em log1p(count).
    print_results : bool
        Se True imprime no stdout (útil em scripts). Caso contrário, silencioso.

    Returns
    -------
    dict
        Estrutura com baselines, métricas, ganhos, bins, topk e meta.
    """
    train_files, train_counts = read_labels_csv(train_dir, labels_filename)
    val_files, val_counts = read_labels_csv(val_dir, labels_filename)

    baselines = compute_simple_baselines(val_counts, train_counts)

    model = None
    preds: Optional[np.ndarray] = None
    metrics: Dict[str, float] = {}
    gains: Dict[str, float] = {}
    bins: list = []
    topk: list = []

    if model_path is not None:
        custom_objects = {}
        try:
            from src.models.mobileNetv2 import CountSumLayer  # type: ignore
            custom_objects['CountSumLayer'] = CountSumLayer
        except Exception:
            pass
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False, compile=False)
        # (Opcional) recompila sem métricas específicas caso não seja necessário
        try:
            model.compile()
        except Exception:
            pass
        preds = predict_model(model, val_files, batch_size=batch_size, img_size=img_size)
        if log_space:
            preds = np.expm1(np.clip(preds, 0, 20))
        if scale_output != 1.0:
            preds = preds * scale_output
        metrics = evaluate_metrics(val_counts, preds)
        # Ganhos
        for base_name in ['mae_zero', 'mae_mean', 'mae_median']:
            gains[f'gain_vs_{base_name}'] = float((baselines[base_name] - metrics['mae']) / baselines[base_name])
        bins = error_by_quantile_bins(val_counts, preds)
        topk = top_k_errors(val_counts, preds, val_files, k=top_k)

    result = {
        'baselines': baselines,
        'metrics': metrics,
        'gains': gains,
        'bins': bins,
        'topk': topk,
        'meta': {
            'model_path': str(model_path) if model_path else None,
            'log_space': log_space,
            'val_size': int(val_counts.shape[0]),
            'train_size': int(train_counts.shape[0]),
            'scale_output': scale_output
        }
    }

    if print_results:
        print('=== BASELINES ===')
        for k, v in baselines.items():
            print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')
        if model_path is None:
            print('\n(Sem modelo)')
        else:
            print('\n=== MÉTRICAS MODELO ===')
            for k, v in metrics.items():
                print(f'{k}: {v:.4f}')
            print('\n=== GANHOS ===')
            for k, v in gains.items():
                print(f'{k}: {v*100:.2f}%')
            print('\n=== BINS ===')
            for b in bins:
                r0, r1 = b['range']
                print(f"bin {b['bin']}: [{r0:.0f}-{r1:.0f}] n={b['n']:4d} mae={b['mae']:.2f}")
            print(f"\n=== TOP {top_k} ===")
            for i, item in enumerate(topk, 1):
                print(f"{i:02d}. err={item['abs_err']:.2f} true={item['y_true']:.0f} pred={item['y_pred']:.0f} file={Path(item['filepath']).name}")

    return result


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Avaliar baselines e modelo de crowd counting.')
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--val-dir', type=Path, required=True)
    parser.add_argument('--labels-filename', type=str, default='labels.csv')
    parser.add_argument('--model-path', type=Path, default=None, help='Caminho para modelo .keras (opcional)')
    parser.add_argument('--log-space', action='store_true', help='Indica que o modelo foi treinado prevendo log1p(count).')
    parser.add_argument('--scale-output', type=float, default=1.0, help='Multiplica a saída do modelo (usar se treinado com alvo escalado).')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    train_files, train_counts = read_labels_csv(args.train_dir, args.labels_filename)
    val_files, val_counts = read_labels_csv(args.val_dir, args.labels_filename)

    # Baselines
    baselines = compute_simple_baselines(val_counts, train_counts)
    print('=== BASELINES ===')
    for k, v in baselines.items():
        print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')

    if args.model_path is None:
        print('\nNenhum modelo fornecido (--model-path). Encerrando após baselines.')
        return

    # Carrega modelo
    print('\nCarregando modelo...')
    custom_objects = {}
    # Tenta importar CountSumLayer se existir
    try:
        from src.models.mobileNetv2 import CountSumLayer  # type: ignore
        custom_objects['CountSumLayer'] = CountSumLayer
    except Exception:
        pass

    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, safe_mode=False, compile=False)

    # Se modelo foi salvo com compile e queremos usar métricas internas
    try:
        model.compile()
    except Exception:
        pass

    print('Gerando predições...')
    preds = predict_model(model, val_files, batch_size=args.batch_size, img_size=tuple(args.img_size))

    if args.log_space:
        preds = np.expm1(np.clip(preds, 0, 20))
    if args.scale_output != 1.0:
        preds = preds * args.scale_output

    # Métricas
    metrics = evaluate_metrics(val_counts, preds)
    print('\n=== MÉTRICAS MODELO ===')
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')

    # Ganhos sobre baselines
    print('\n=== GANHO SOBRE BASELINES (relativo) ===')
    for base_name in ['mae_zero', 'mae_mean', 'mae_median']:
        gain = (baselines[base_name] - metrics['mae']) / baselines[base_name]
        print(f'ganho_vs_{base_name}: {gain*100:.2f}%')

    # Bins
    bins = error_by_quantile_bins(val_counts, preds)
    print('\n=== MAE POR BINS (quantis) ===')
    for b in bins:
        r0, r1 = b['range']
        print(f"bin {b['bin']}: [{r0:.0f}-{r1:.0f}] n={b['n']:4d} mae={b['mae']:.2f}")

    # Top-K
    topk = top_k_errors(val_counts, preds, val_files, k=args.top_k)
    print(f"\n=== TOP {args.top_k} MAIORES ERROS ABSOLUTOS ===")
    for i, item in enumerate(topk, 1):
        print(f"{i:02d}. err={item['abs_err']:.2f} true={item['y_true']:.0f} pred={item['y_pred']:.0f} file={Path(item['filepath']).name}")

    print('\nConcluído.')


if __name__ == '__main__':
    main()
