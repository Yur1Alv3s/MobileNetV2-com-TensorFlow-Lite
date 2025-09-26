#!/usr/bin/env python3
# Crowd_Counting_Problem/strip_exif.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

try:
    import piexif  # remoção EXIF lossless
    _PIEXIF_OK = True
except Exception:
    _PIEXIF_OK = False

from PIL import Image  # usado para checar se existe EXIF e (fallback opcional)

# ------------------------------- Utils --------------------------------

@dataclass
class Result:
    path: Path
    action: str     # "removed", "no_exif", "skipped_ext", "error", "reencoded"
    message: str = ""


def expand_inputs(args_paths: List[str], from_file: str | None, globs: List[str]) -> List[Path]:
    """
    expand_inputs(args_paths, from_file, globs) -> list[Path]

    Coleta e expande os caminhos informados (args diretos, --from-file, --glob),
    removendo duplicatas e mantendo a ordem.
    Entrada:
      - args_paths: lista de caminhos passados diretamente.
      - from_file: caminho para um .txt com um caminho por linha (opcional).
      - globs: padrões glob (ex.: "data/val/images/*.jpg").
    Saída:
      - lista de Paths existentes.
    """
    seen = set()
    out: List[Path] = []

    def _add(p: Path) -> None:
        if p.exists() and p.is_file():
            if p not in seen:
                out.append(p); seen.add(p)

    for s in args_paths:
        _add(Path(s))

    if from_file:
        for line in Path(from_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                _add(Path(line))

    # glob manual (para manter compat no Windows/terminal)
    import glob as _glob
    for pat in globs:
        for s in _glob.glob(pat):
            _add(Path(s))

    return out


def has_exif(p: Path) -> bool:
    """
    has_exif(p) -> bool

    Retorna True se a imagem possui bloco EXIF (PIL getexif() com tamanho > 0).
    Entrada:
      - p: caminho do arquivo de imagem.
    Saída:
      - bool indicando presença de EXIF.
    """
    try:
        with Image.open(p) as im:
            ex = im.getexif()
            return bool(ex and len(ex) > 0)
    except Exception:
        # Se nem abrir, deixa para o fluxo principal reportar como erro
        return False


def strip_exif_lossless(p: Path) -> None:
    """
    strip_exif_lossless(p) -> None

    Remove o EXIF **sem re-encodar** (lossless) usando piexif.
    Entrada:
      - p: caminho do arquivo .jpg/.jpeg/.tif/.tiff
    Saída:
      - None (efeito colateral: arquivo atualizado in-place)
    """
    piexif.remove(str(p))


def reencode_without_exif(p: Path, quality: int = 95, subsampling: int = 0) -> None:
    """
    reencode_without_exif(p, quality=95, subsampling=0) -> None

    Fallback opcional: re-encoda a imagem **sem EXIF** (pode alterar levemente pixels).
    Entrada:
      - p: caminho do arquivo
      - quality: qualidade JPEG (95 recomendado)
      - subsampling: 0 = 4:4:4 (melhor qualidade)
    Saída:
      - None (arquivo sobrescrito de forma atômica)
    """
    tmp = p.with_suffix(p.suffix + ".tmp")
    with Image.open(p) as im:
        im = im.convert("RGB")
        im.save(tmp, "JPEG", quality=quality, subsampling=subsampling, optimize=True)
    tmp.replace(p)


def is_supported_ext(p: Path) -> bool:
    """
    is_supported_ext(p) -> bool

    Verifica se a extensão é típica de arquivo com EXIF (JPEG/TIFF).
    Entrada:
      - p: caminho do arquivo
    Saída:
      - True para { .jpg, .jpeg, .tif, .tiff } (case-insensitive)
    """
    return p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"}


# ------------------------------- Main ---------------------------------

def process(paths: Iterable[Path], allow_reencode: bool, quality: int) -> List[Result]:
    """
    process(paths, allow_reencode, quality) -> list[Result]

    Processa cada imagem: se tiver EXIF, remove (lossless com piexif).
    Se piexif indisponível e allow_reencode=True, re-encoda sem EXIF.

    Entrada:
      - paths: iterável de caminhos de imagem.
      - allow_reencode: se True, usa fallback de re-encode quando piexif não estiver disponível.
      - quality: qualidade JPEG para o fallback (95 recomendado).
    Saída:
      - lista de Result com status por arquivo.
    """
    results: List[Result] = []
    for p in paths:
        if not is_supported_ext(p):
            results.append(Result(p, "skipped_ext", "extensão não suportada para EXIF"))
            continue

        try:
            if not has_exif(p):
                results.append(Result(p, "no_exif", "sem EXIF"))
                continue

            if _PIEXIF_OK:
                strip_exif_lossless(p)
                results.append(Result(p, "removed", "EXIF removido (lossless)"))
            else:
                if allow_reencode:
                    reencode_without_exif(p, quality=quality, subsampling=0)
                    results.append(Result(p, "reencoded", "EXIF removido via re-encode"))
                else:
                    results.append(Result(
                        p, "error",
                        "piexif não instalado; rode 'pip install piexif' ou use --reencode"
                    ))
        except Exception as e:
            results.append(Result(p, "error", str(e)))
    return results


def main() -> int:
    """
    main() -> int

    CLI para remover EXIF apenas dos arquivos informados.
    Flags:
      - caminhos diretos (posicionais): arquivos individuais
      - --from-file <txt>: um caminho por linha
      - --glob <padrão>: pode repetir, ex.: --glob 'data/val/images/*.jpg'
      - --reencode: se piexif não estiver disponível, re-encoda sem EXIF (com perdas mínimas)
      - --quality: qualidade JPEG do fallback (default 95)
    Retorno:
      - código de saída 0 (ok) ou 1 (se houve algum erro)
    """
    ap = argparse.ArgumentParser(description="Remover EXIF apenas das imagens informadas")
    ap.add_argument("paths", nargs="*", help="caminhos de arquivos (opcional)")
    ap.add_argument("--from-file", type=str, default=None, help="arquivo .txt com um caminho por linha")
    ap.add_argument("--glob", action="append", default=[], help="padrões glob (pode repetir)")
    ap.add_argument("--reencode", action="store_true", help="usar fallback de re-encode se piexif não estiver disponível")
    ap.add_argument("--quality", type=int, default=95, help="qualidade JPEG para fallback de re-encode")
    args = ap.parse_args()

    files = expand_inputs(args.paths, args.from_file, args.glob)
    if not files:
        print("Nenhum arquivo encontrado. Informe caminhos, --from-file ou --glob.", file=sys.stderr)
        return 1

    res = process(files, allow_reencode=args.reencode, quality=args.quality)

    # resumo
    counts = {}
    for r in res:
        counts[r.action] = counts.get(r.action, 0) + 1

    print("Resumo:")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")

    # exemplos de erros (se houver)
    errs = [r for r in res if r.action == "error"]
    if errs:
        print("\nErros (até 10):")
        for r in errs[:10]:
            print(" -", r.path, "->", r.message)

    # imprime quais foram alteradas (removed/reencoded)
    changed = [r for r in res if r.action in ("removed", "reencoded")]
    if changed:
        print("\nAlteradas:")
        for r in changed[:10]:
            print(" -", r.path, "->", r.action, ":", r.message)
        if len(changed) > 10:
            print(f"  ... (+{len(changed)-10} outras)")

    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
