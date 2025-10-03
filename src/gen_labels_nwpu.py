#!/usr/bin/env python3

"""
python Crowd_Counting_Problem/data/gen_labels_nwpu.py \
        --dataset-root "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/The NWPU-Crowd Dataset" \
        --output-root "/home/yuri-alves/Área de Trabalho/VScode/TCC/Codigo/src/data/nwpu_regression" \
  --balanced \
  --balance-bins 10
"""
"""Gera arquivos labels.csv (filename,value) para o dataset NWPU Crowd
criando (por padrão via symlink) uma estrutura compatível com o loader
`load_dataset_regression`.

Estrutura de saída (exemplo):
    output_root/
        train/
            labels.csv
            0001.jpg -> (symlink para .../images_partX/0001.jpg)
            ...
        val/
            labels.csv
        test/ (opcional com --include-test)
            labels.csv

O CSV fica no formato:
    filename,value
    0001.jpg,45
    0002.jpg,13

Uso:
    python gen_labels_nwpu.py \
        --dataset-root "/caminho/The NWPU-Crowd Dataset" \
        --output-root ./nwpu_regression \
        --link         # (ou --copy)

Opções:
    --copy            Copia as imagens ao invés de criar symlinks.
    --include-test    Também gera split de teste (se quiser avaliação offline).
    --limit N         Limita número de imagens por split (debug).

Requisitos: apenas biblioteca padrão (stdlib) + Python >=3.8
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import csv
import shutil
import random

# -------------------------- Utilidades -------------------------- #

def read_json_counts(json_dir: Path) -> Dict[str, int]:
    """Lê todos os JSONs e retorna dict id -> human_num.

    JSON tem formato: {"img_id": "0001.jpg", "human_num": 45, ...}
    Chave retornada será SEM a extensão (ex: '0001').
    """
    mapping: Dict[str, int] = {}
    json_files = sorted(json_dir.glob('*.json'))
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            img_id = data.get('img_id')  # ex: '0001.jpg'
            human_num = data.get('human_num')
            if img_id is None or human_num is None:
                print(f"[WARN] Campos faltando em {jf}")
                continue
            stem = Path(img_id).stem  # '0001'
            mapping[stem] = int(human_num)
        except Exception as e:
            print(f"[ERRO] Falha lendo {jf}: {e}")
    print(f"[INFO] JSONs processados: {len(mapping)} entradas")
    return mapping


def read_split_ids(split_file: Path) -> List[str]:
    """Lê arquivo train.txt / val.txt / test.txt.

    Cada linha começa com um ID (ex: '0001 1 2'). Pegamos só a primeira coluna.
    """
    ids: List[str] = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 0:
                continue
            ids.append(parts[0])
    return ids


def locate_image(image_id: str, image_parts_dirs: Iterable[Path]) -> Path | None:
    """Procura arquivo <image_id>.jpg nos diretórios de imagem (images_part*)."""
    filename = f"{image_id}.jpg"
    for d in image_parts_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except OSError:
            # fallback: copy
            shutil.copy2(src, dst)


def write_labels_csv(out_dir: Path, rows: List[Tuple[str, int]]):
    csv_path = out_dir / 'labels.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'value'])
        writer.writerows(rows)
    print(f"[OK] Gerado {csv_path} ({len(rows)} linhas)")


def build_split(
    split_name: str,
    ids: List[str],
    counts: Dict[str, int],
    image_parts_dirs: List[Path],
    out_root: Path,
    copy: bool = False,
    limit: int | None = None,
) -> List[Tuple[str, int]]:
    """Constrói um split (train/val/test).

    Para cada ID:
        - localiza imagem
        - obtém contagem (se não existir, pula e avisa)
        - cria symlink/cópia
        - acumula linha para labels.csv
    """
    if limit is not None:
        ids = ids[:limit]

    out_dir = out_root / split_name
    ensure_dir(out_dir)

    rows: List[Tuple[str, int]] = []
    missing_count = 0
    missing_img = 0

    for idx, image_id in enumerate(ids, 1):
        if image_id not in counts:
            print(f"[WARN][{split_name}] Sem count para {image_id}")
            missing_count += 1
            continue
        img_path = locate_image(image_id, image_parts_dirs)
        if img_path is None:
            print(f"[WARN][{split_name}] Imagem não encontrada para {image_id}")
            missing_img += 1
            continue
        # destino (mantém nome original <id>.jpg)
        dst = out_dir / img_path.name
        symlink_or_copy(img_path, dst, copy=copy)
        rows.append((img_path.name, counts[image_id]))
        if idx % 1000 == 0:
            print(f"[INFO][{split_name}] Processados {idx}/{len(ids)}")

    write_labels_csv(out_dir, rows)
    if missing_count or missing_img:
        print(f"[RESUMO][{split_name}] sem_count={missing_count} sem_imagem={missing_img} válidas={len(rows)}")
    return rows


def build_balanced_subset(
    base_split_name: str,
    rows: List[Tuple[str, int]],
    out_root: Path,
    image_parts_dirs: List[Path],
    copy: bool,
    bins: int,
    samples_per_bin: Optional[int],
    seed: int,
) -> None:
    """Cria um subset balanceado (por faixas de contagem) a partir de rows.

    Estratégia:
        - Ordena counts
        - Cria 'bins' quantis aproximados
        - Agrupa amostras por bin
        - Seleciona 'samples_per_bin' amostras de cada (ou mínimo disponível se não fornecido)
        - Gera nova pasta <split>_balanced com labels.csv
    """
    if not rows:
        print(f"[BALANCE] Nenhuma linha para balancear em {base_split_name}")
        return

    random.seed(seed)
    counts_only = sorted([c for _, c in rows])
    n = len(counts_only)
    if bins < 2:
        print("[BALANCE] Número de bins < 2; ignorando.")
        return

    # Determinar limites (quantis) - lista de thresholds (exclui max)
    thresholds: List[int] = []
    for i in range(1, bins):
        idx = min(n - 1, int(n * i / bins))
        thresholds.append(counts_only[idx])

    def assign_bin(value: int) -> int:
        for bi, thr in enumerate(thresholds):
            if value <= thr:
                return bi
        return len(thresholds)  # último bin

    bin_buckets: Dict[int, List[Tuple[str, int]]] = {i: [] for i in range(bins)}
    for fn, c in rows:
        b = assign_bin(c)
        bin_buckets[b].append((fn, c))

    # Remover bins vazios
    non_empty_bins = {k: v for k, v in bin_buckets.items() if v}
    if not non_empty_bins:
        print("[BALANCE] Todos os bins vazios?! Abortando.")
        return

    if samples_per_bin is None:
        samples_per_bin = min(len(v) for v in non_empty_bins.values())
    else:
        # Garantir que não exceda menor bin
        samples_per_bin = min(samples_per_bin, *(len(v) for v in non_empty_bins.values()))

    balanced_rows: List[Tuple[str, int]] = []
    for bi, bucket in non_empty_bins.items():
        random.shuffle(bucket)
        take = bucket[:samples_per_bin]
        balanced_rows.extend(take)
        print(f"[BALANCE] Bin {bi}: total={len(bucket)} usando={len(take)}")

    # Criar saída
    out_dir = out_root / f"{base_split_name}_balanced"
    ensure_dir(out_dir)

    # Precisamos saber origem das imagens. Vamos localizar de novo (custo aceitável)
    def find_src(fn: str) -> Optional[Path]:
        stem = Path(fn).stem
        # Usa locate_image com id sem extensão
        return locate_image(stem, image_parts_dirs)

    for fn, _ in balanced_rows:
        src = find_src(fn)
        if src is None:
            print(f"[BALANCE][WARN] Não achou imagem para {fn}")
            continue
        dst = out_dir / fn
        symlink_or_copy(src, dst, copy=copy)

    write_labels_csv(out_dir, balanced_rows)
    print(f"[BALANCE] Subset gerado em {out_dir} (total {len(balanced_rows)})")


# -------------------------- Main -------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Gera labels.csv para NWPU Crowd (regressão de contagem)")
    p.add_argument('--dataset-root', type=Path, required=True,
                   help="Diretório raiz do NWPU (aquele que contém images_part1, jsons, train.txt, etc)")
    p.add_argument('--output-root', type=Path, required=True,
                   help="Diretório onde será criada a nova estrutura (train/, val/, ...)")
    # Nota: o comportamento atual é COPIAR as imagens (sem usar symlinks)
    # A opção de criar symlinks foi removida por segurança/reprodutibilidade.
    p.add_argument('--include-test', action='store_true', help="Também gerar split de teste")
    p.add_argument('--limit', type=int, default=None, help="Limitar número de imagens por split (debug)")
    # Balanceamento
    p.add_argument('--balanced', action='store_true', help="Gerar subset balanceado por bins de contagem (aplica-se apenas ao treino por padrão)")
    p.add_argument('--balance-bins', type=int, default=10, help="Número de bins (quantis) para balancear")
    p.add_argument('--balance-samples-per-bin', type=int, default=None, help="Forçar número de amostras por bin (default=min bin)")
    p.add_argument('--balance-seed', type=int, default=42, help="Seed para amostragem aleatória no balanceamento")
    p.add_argument('--balance-include-val', action='store_true', help='Também gerar subset balanceado para validação (val_balanced). Por padrão só gera train_balanced.')
    return p.parse_args()


def main():
    args = parse_args()
    dataset_root: Path = args.dataset_root
    out_root: Path = args.output_root
    # Forçar cópia das imagens (não usar symlinks)
    copy_flag: bool = True

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root não existe: {dataset_root}")

    json_dir = dataset_root / 'jsons'
    if not json_dir.exists():
        raise SystemExit(f"Pasta de JSONs não encontrada: {json_dir}")

    # Coletar pastas images_part*
    image_parts_dirs = sorted([p for p in dataset_root.glob('images_part*') if p.is_dir()])
    if not image_parts_dirs:
        raise SystemExit("Nenhuma pasta images_part* encontrada")
    print(f"[INFO] Encontradas {len(image_parts_dirs)} partições de imagens")

    counts = read_json_counts(json_dir)

    train_ids = read_split_ids(dataset_root / 'train.txt')
    val_ids = read_split_ids(dataset_root / 'val.txt')
    test_ids = read_split_ids(dataset_root / 'test.txt') if args.include_test else []

    print(f"[INFO] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    ensure_dir(out_root)

    train_rows = build_split('train', train_ids, counts, image_parts_dirs, out_root, copy=copy_flag, limit=args.limit)
    val_rows = build_split('val', val_ids, counts, image_parts_dirs, out_root, copy=copy_flag, limit=args.limit)
    if args.include_test:
        test_rows = build_split('test', test_ids, counts, image_parts_dirs, out_root, copy=copy_flag, limit=args.limit)
    else:
        test_rows = []

    if args.balanced:
        print("[BALANCE] Gerando subset balanceado para o TREINO...")
        build_balanced_subset('train', train_rows, out_root, image_parts_dirs, copy_flag,
                              bins=args.balance_bins,
                              samples_per_bin=args.balance_samples_per_bin,
                              seed=args.balance_seed)
        if args.balance_include_val:
            print("[BALANCE] --balance-include-val fornecido: gerando subset balanceado para VALIDAÇÃO...")
            build_balanced_subset('val', val_rows, out_root, image_parts_dirs, copy_flag,
                                  bins=args.balance_bins,
                                  samples_per_bin=args.balance_samples_per_bin,
                                  seed=args.balance_seed)
        if args.include_test and test_rows:
            print("[BALANCE] Gerando subset balanceado para TEST (porque --include-test está ativo)...")
            build_balanced_subset('test', test_rows, out_root, image_parts_dirs, copy_flag,
                                  bins=args.balance_bins,
                                  samples_per_bin=args.balance_samples_per_bin,
                                  seed=args.balance_seed)

    print("[DONE] Geração concluída.")
    print(f"Para treinar: use train_dir={out_root / 'train'} val_dir={out_root / 'val'}")


if __name__ == '__main__':
    main()
