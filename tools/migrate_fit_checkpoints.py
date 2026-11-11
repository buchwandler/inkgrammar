from __future__ import annotations

from pathlib import Path

from inkgrammar.fitting.checkpoint import load_checkpoint, save_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / "data" / "training" / "sas-1968" / "checkpoints"
    paths = sorted(checkpoint_dir.glob("*.json"))
    if len(paths) != 4:
        raise SystemExit(f"expected four SAS primitive checkpoints, found {len(paths)}")
    for path in paths:
        save_checkpoint(load_checkpoint(path), path)
        print(path.relative_to(root))


if __name__ == "__main__":
    main()
