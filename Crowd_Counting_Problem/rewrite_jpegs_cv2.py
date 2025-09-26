from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple
import argparse
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")

def iter_imgs(splits: Iterable[str]) -> Iterable[Path]:
    for split in splits:
        folder = DATA / split / "images"
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.suffix in EXTS and p.is_file():
                yield p

def rewrite_with_cv2(p: Path, quality: int = 95) -> None:
    data = np.fromfile(str(p), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode retornou None (arquivo muito corrompido?)")
    # temp com extensão .jpg (writer reconhece)
    tmp = p.with_name(p.stem + ".__tmp__.jpg")
    ok = cv2.imwrite(str(tmp), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imwrite falhou (não conseguiu escrever .jpg)")
    tmp.replace(p)

def process(paths: Iterable[Path], quality: int = 95) -> Tuple[int, int, List[Tuple[Path, str]]]:
    ok = fail = 0
    fails: List[Tuple[Path, str]] = []
    for p in paths:
        try:
            rewrite_with_cv2(p, quality=quality)
            ok += 1
        except Exception as e:
            fail += 1
            fails.append((p, str(e)))
    return ok, fail, fails

def main() -> int:
    ap = argparse.ArgumentParser(description="Regravar JPEGs com OpenCV para higienizar arquivos")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--only-file", type=str, default=None, help="TXT com um caminho de imagem por linha")
    args = ap.parse_args()

    if args.only_file:
        paths = [Path(line.strip()) for line in Path(args.only_file).read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.strip().startswith("#")]
    else:
        paths = list(iter_imgs(args.splits))

    if not paths:
        print("Nenhuma imagem encontrada para processar.")
        return 1

    print(f"Processando {len(paths)} imagens (qualidade={args.quality})...")
    ok, fail, fails = process(paths, quality=args.quality)
    print(f"Regravadas com sucesso: {ok} | Falhas: {fail}")
    if fails:
        print("Exemplos de falhas (até 10):")
        for p, e in fails[:10]:
            print(" -", p, "->", e)
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
