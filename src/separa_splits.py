#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera os splits (train/val/test) a partir de uma pasta de imagens e um arquivo
de rótulos por intervalo. Move ou copia as imagens para pastas organizadas por
split e classe (fire/nofire).

Observação: o script assume que os nomes dos arquivos têm um número de frame
que pode ser extraído (ex.: "Frame (123).jpg" ou qualquer nome cujo último
grupo numérico represente o frame). Não altera o conteúdo das imagens.
"""

import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configurações principais
SOURCE_IMAGES_DIR = Path("data/images")       # Pasta com todas as imagens
LABELS_FILE       = Path("data/labels.txt")   # Arquivo com rótulos por intervalo
OUTPUT_DIR        = Path("data/splits")       # Saída: train/val/test/{fire,nofire}

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}  # deve somar 1.0

COPY_INSTEAD_OF_MOVE = True   # True = copia os arquivos; False = move

BALANCE_TRAIN = True
BALANCE_VAL   = True
BALANCE_TEST  = False  # no teste, normalmente mantemos a distribuição real

RANDOM_SEED = 42

ACCEPTED_EXTS = {"jpg", "jpeg", "png", "bmp"}

# Filtra nomes de arquivos que contenham estas substrings (case-insensitive).
# Ex.: para pegar apenas RGB: ["RGB"]. Para não filtrar: [].
REQUIRE_SUBSTRINGS = ["RGB"]

# =====================================================

def map_label_to_binary(label: str) -> str:
    """Converte rótulos YY/YN/NN/NY em classes binárias: fire/nofire."""
    # YY, YN -> fire ; NN, NY -> nofire
    label = label.upper()
    if label in {"YY", "YN"}:
        return "fire"
    elif label in {"NN", "NY"}:
        return "nofire"
    else:
        return "ignore"

def parse_labels_file(path: Path):
    """Lê o arquivo de rótulos e retorna intervalos válidos com classe binária."""
    intervals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not re.match(r"^\d", line):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            try:
                start = int(parts[0]); end = int(parts[1]); raw = parts[2].upper()
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            b = map_label_to_binary(raw)
            if b == "ignore":
                continue
            intervals.append({"start": start, "end": end, "raw": raw, "bin": b})
    intervals.sort(key=lambda x: (x["bin"], x["start"]))
    return intervals

# Indexador para nomes do tipo "... Frame (123).jpg" (ou com número no fim)

PAREN_NUM_RE   = re.compile(r"\((\d+)\)")           # pega número entre parênteses
LAST_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")       # pega último grupo numérico na string

def extract_frame_number(filename: str) -> int | None:
    """Tenta extrair o número do frame do nome do arquivo.
    Ordem de busca:
      1) número entre parênteses: "Frame (123).jpg" -> 123
      2) último número do nome:   "Frame 000123.jpg" -> 123
    Retorna int ou None se não houver número.
    """
    m = PAREN_NUM_RE.search(filename)
    if m:
        return int(m.group(1))
    m = LAST_NUMBER_RE.search(filename)
    if m:
        return int(m.group(1))
    return None

def index_numeric_images(src_dir: Path):
    """Cria um índice {frame_number: Path} a partir dos nomes.
    - Filtra pelas extensões em ACCEPTED_EXTS
    - Extrai o número com extract_frame_number()
    - (Opcional) exige substrings (REQUIRE_SUBSTRINGS), ex.: "RGB"
    Em caso de duplicata (mesmo frame com extensões diferentes), mantém o primeiro.
    """
    idx = {}
    reqs = [s.lower() for s in REQUIRE_SUBSTRINGS]
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext not in ACCEPTED_EXTS:
            continue
        name_lower = p.name.lower()
        if reqs and not all(s in name_lower for s in reqs):
            continue
        n = extract_frame_number(p.name)
        if n is None:
            continue
        if n not in idx:  # mantém o primeiro; ajuste aqui se quiser preferir .jpg, etc.
            idx[n] = p
    return idx

# ----------------------------------------------------

def split_intervals(intervals, ratios, seed=42):
    """Divide por intervalos, estratificando por classe binária.
    Caso uma classe tenha só 1 intervalo, ele é subdividido proporcionalmente
    entre train/val/test para garantir que ambas as classes apareçam nos três splits.
    """
    import random
    from collections import defaultdict

    random.seed(seed)
    by_cls = defaultdict(list)
    for it in intervals:
        by_cls[it["bin"]].append(it)

    splits = {"train": [], "val": [], "test": []}
    names = ["train", "val", "test"]

    for cls, lst in by_cls.items():
        # Ordena por início pra termos cortes coerentes
        lst = sorted(lst, key=lambda x: x["start"])

        total_frames = sum(i["end"] - i["start"] + 1 for i in lst)
        targets_float = {k: total_frames * ratios[k] for k in names}

        if len(lst) == 1:
            # --- CASO ESPECIAL: apenas 1 intervalo -> subdivide proporcionalmente ---
            it = lst[0]
            size = it["end"] - it["start"] + 1

            # Tamanhos inteiros aproximados
            train_sz = int(size * ratios["train"])
            val_sz   = int(size * ratios["val"])
            test_sz  = size - train_sz - val_sz

            # Garante ao menos 1 frame em val/test quando possível
            if size >= 3:
                if val_sz == 0:
                    val_sz += 1
                    train_sz = max(0, train_sz - 1)
                if test_sz == 0:
                    test_sz += 1
                    train_sz = max(0, train_sz - 1)

            s = it["start"]
            if train_sz > 0:
                splits["train"].append({"start": s, "end": s + train_sz - 1, "raw": it["raw"], "bin": cls})
                s += train_sz
            if val_sz > 0:
                splits["val"].append({"start": s, "end": s + val_sz - 1, "raw": it["raw"], "bin": cls})
                s += val_sz
            if test_sz > 0:
                splits["test"].append({"start": s, "end": it["end"], "raw": it["raw"], "bin": cls})
            continue

        # --- CASO GERAL: vários intervalos -> greedy por maior déficit ---
        random.shuffle(lst)
        acc = {k: 0 for k in names}
        for itv in lst:
            size = itv["end"] - itv["start"] + 1
            deficits = {k: targets_float[k] - acc[k] for k in names}
            bucket = max(deficits, key=deficits.get)
            splits[bucket].append(itv)
            acc[bucket] += size

        # Garantia mínima: evita split vazio para essa classe
        for k in names:
            if not any(x["bin"] == cls for x in splits[k]):
                # Move um pedacinho do maior bucket dessa classe
                donor = max(
                    names,
                    key=lambda kk: sum(x["end"] - x["start"] + 1 for x in splits[kk] if x["bin"] == cls)
                )
                # pega o maior intervalo do doador
                idx, donor_it = max(
                    [(i, x) for i, x in enumerate(splits[donor]) if x["bin"] == cls],
                    key=lambda t: t[1]["end"] - t[1]["start"] + 1
                )
                dsize = donor_it["end"] - donor_it["start"] + 1
                move = max(1, dsize // 10)  # corta ~10% do final
                new_end = donor_it["end"]
                new_start = new_end - move + 1
                # encurta o doador
                splits[donor][idx]["end"] = new_start - 1
                # cria subintervalo no split carente
                splits[k].append({"start": new_start, "end": new_end, "raw": donor_it["raw"], "bin": cls})

    return splits

def ensure_dirs(base_out: Path):
    """Garante a criação das pastas de saída (train/val/test e suas classes)."""
    for part in ["train", "val", "test"]:
        (base_out / part / "fire").mkdir(parents=True, exist_ok=True)
        (base_out / part / "nofire").mkdir(parents=True, exist_ok=True)

def copy_or_move(src: Path, dst: Path, copy=True):
    """Copia ou move um arquivo, criando a pasta de destino se precisar."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

def place_frames_balanced(intervals, img_index, out_dir: Path, split_name: str,
                          copy=True, balance=True, seed=42):
    """Seleciona frames dos intervalos e coloca nas pastas do split.
    Se balance=True, limita para ter o mesmo número de fire/nofire.
    """
    random.seed(seed)
    fire_frames, nofire_frames = [], []

    for itv in intervals:
        if itv["bin"] not in {"fire", "nofire"}:
            continue
        tgt = fire_frames if itv["bin"] == "fire" else nofire_frames
        # só conta frames existentes
        tgt.extend([n for n in range(itv["start"], itv["end"] + 1) if n in img_index])

    random.shuffle(fire_frames)
    random.shuffle(nofire_frames)

    if balance:
        k = min(len(fire_frames), len(nofire_frames))
        fire_frames   = fire_frames[:k]
        nofire_frames = nofire_frames[:k]

    counts = {"fire": 0, "nofire": 0, "missing": 0}
    for n in fire_frames:
        src = img_index.get(n)
        if src is None:
            counts["missing"] += 1; continue
        dst = out_dir / split_name / "fire" / f"{n}{src.suffix.lower()}"
        copy_or_move(src, dst, copy=copy)
        counts["fire"] += 1

    for n in nofire_frames:
        src = img_index.get(n)
        if src is None:
            counts["missing"] += 1; continue
        dst = out_dir / split_name / "nofire" / f"{n}{src.suffix.lower()}"
        copy_or_move(src, dst, copy=copy)
        counts["nofire"] += 1

    return counts

def summarize_counts(title, c):
    """Resumo amigável de contagens por classe e total."""
    total = c["fire"] + c["nofire"]
    ratio = (f"{(c['fire']/total*100):.1f}% fire / {(c['nofire']/total*100):.1f}% nofire"
             if total else "n/a")
    return f"{title}: fire={c['fire']:,} | nofire={c['nofire']:,} | total={total:,} | {ratio}"

def main():
    """Ponto de entrada do script: valida configurações, cria splits e imprime um resumo."""
    s = sum(SPLIT_RATIOS.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"As razões devem somar 1.0, somam {s}")

    print(">> Indexando imagens...")
    img_index = index_numeric_images(SOURCE_IMAGES_DIR)
    if not img_index:
        raise FileNotFoundError(f"Nenhuma imagem elegível encontrada em {SOURCE_IMAGES_DIR}")
    print(f">> Imagens indexadas: {len(img_index):,}")

    print(">> Lendo intervalos...")
    intervals = parse_labels_file(LABELS_FILE)
    if not intervals:
        raise ValueError("Nenhum intervalo válido encontrado (YY/YN/NN/NY).")

    total_fire   = sum(i["end"] - i["start"] + 1 for i in intervals if i["bin"] == "fire")
    total_nofire = sum(i["end"] - i["start"] + 1 for i in intervals if i["bin"] == "nofire")
    print(f">> Frames (binário): fire={total_fire:,} | nofire={total_nofire:,} | total={total_fire+total_nofire:,}")

    print(">> Split por INTERVALOS (estratificado por classe binária)...")
    splits = split_intervals(intervals, SPLIT_RATIOS, seed=RANDOM_SEED)

    ensure_dirs(OUTPUT_DIR)

    print(">> Distribuindo arquivos...")
    counts_train = place_frames_balanced(
        splits["train"], img_index, OUTPUT_DIR, "train",
        copy=COPY_INSTEAD_OF_MOVE, balance=BALANCE_TRAIN, seed=RANDOM_SEED
    )
    counts_val = place_frames_balanced(
        splits["val"], img_index, OUTPUT_DIR, "val",
        copy=COPY_INSTEAD_OF_MOVE, balance=BALANCE_VAL, seed=RANDOM_SEED
    )
    counts_test = place_frames_balanced(
        splits["test"], img_index, OUTPUT_DIR, "test",
        copy=COPY_INSTEAD_OF_MOVE, balance=BALANCE_TEST, seed=RANDOM_SEED
    )

    print("\n===== RESUMO =====")
    print(summarize_counts("train", counts_train))
    print(summarize_counts("val  ", counts_val))
    print(summarize_counts("test ", counts_test))

    if COPY_INSTEAD_OF_MOVE:
        print("\nModo: COPIAR (origem mantida).")
    else:
        print("\nModo: MOVER (origem removida).")

if __name__ == "__main__":
    main()
