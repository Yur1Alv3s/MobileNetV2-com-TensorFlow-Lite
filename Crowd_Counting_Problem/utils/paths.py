from pathlib import Path
import yaml

# Raiz do projeto (CROWD_COUNTING/)
ROOT = Path(__file__).resolve().parents[1]

# Pastas principais
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

def list_file(split: str) -> Path:
    return DATA / "lists" / f"{split}.txt"

def split_dir(split: str) -> Path:
    return DATA / split

def img_dir(split: str) -> Path:
    return split_dir(split) / "images"

def ann_json_dir(split: str) -> Path:
    return split_dir(split) / "annotations" / "points_json"

def ann_mat_dir(split: str) -> Path:
    return split_dir(split) / "annotations" / "mats"

def density_dir(split: str) -> Path:
    return split_dir(split) / "density_maps"

def ensure_artifact_dirs() -> None:
    for p in [
        ARTIFACTS,
        ARTIFACTS / "models",
        ARTIFACTS / "metrics",
        ARTIFACTS / "benchmarks",
        ARTIFACTS / "logs",
        ARTIFACTS / "previews",
        ARTIFACTS / "predict_test",
    ]:
        p.mkdir(parents=True, exist_ok=True)

def load_cfg(path: str | Path | None = None) -> dict:
    """
    load_cfg(path: str|Path|None=None) -> dict

    Carrega o YAML de configuração. Se 'path' for None, usa
    'config/crowd_mnv2_s8.yaml' relativo à raiz do projeto.

    Entrada:
      - path: caminho opcional para o YAML (str/Path) ou None.
    Saída:
      - dict com a configuração, já com defaults preenchidos:
          input_size: [512, 512]
          output_stride: 8
          density: {mode: adaptive, k: 3, beta: 0.3, min_sigma: 1.0}
    Comportamento:
      - Se o YAML estiver vazio, retorna {} e aplica os defaults acima.
      - Se o arquivo não existir, lança FileNotFoundError com dica amigável.
    """
    # Resolve caminho padrão quando None
    if path is None:
        path = ROOT / "config" / "crowd_mnv2_s8.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de config não encontrado: {path}\n"
            f"Dica: crie '{ROOT / 'config' / 'crowd_mnv2_s8.yaml'}' "
            f"ou passe --config /caminho/para/seu.yaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}  # se YAML vazio, vira {}

    # Defaults úteis
    cfg.setdefault("experiment_name", "crowd_mnv2_s8")
    cfg.setdefault("input_size", [512, 512])
    cfg.setdefault("output_stride", 8)
    cfg.setdefault("batch_size", 4)
    cfg.setdefault("epochs", 100)
    cfg.setdefault("lr", 1e-4)
    cfg.setdefault("early_stopping_patience", 10)
    cfg.setdefault("augmentations", {"flip": True})
    cfg.setdefault("density", {"mode": "adaptive", "k": 3, "beta": 0.3, "min_sigma": 1.0})

    return cfg