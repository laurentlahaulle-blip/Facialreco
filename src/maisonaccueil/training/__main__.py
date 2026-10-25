"""Outil en ligne de commande du lot E : ``python -m maisonaccueil.training <commande>``.

- ``manifest --db base.sqlite --out var/e/manifest.json`` : manifeste (lecture seule) ;
- ``check MANIFEST [--disjoint train test]`` : contrôle des fuites + effectifs ;
- ``train --config c.json --manifest m.json --data-root DIR [--runs-root runs]`` ;
- ``export RUN_DIR`` : export ONNX ;
- ``smoke [--runs-root var/e/runs]`` : test de fumée SYNTHÉTIQUE (quelques pas, CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ExperimentConfig
from .manifest import Manifest, manifest_from_database


def _cmd_manifest(args: argparse.Namespace) -> int:
    manifest = manifest_from_database(Path(args.db), include_unassigned=args.include_unassigned)
    manifest.save(Path(args.out))
    print(json.dumps(manifest.counts(), indent=2))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    manifest = Manifest.load(Path(args.manifest))
    if args.disjoint:
        manifest.check_identity_disjoint(*args.disjoint)
    print(json.dumps({"preprocess": manifest.preprocess, "counts": manifest.counts()}, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .data import AlignedTrainingSet
    from .trainer import train

    config = ExperimentConfig.load(Path(args.config))
    manifest = Manifest.load(Path(args.manifest))
    dataset = AlignedTrainingSet.from_manifest(manifest, Path(args.data_root))
    result = train(
        config,
        dataset,
        runs_root=Path(args.runs_root),
        manifest_counts=manifest.counts(),
    )
    print(f"Expérience : {result.run.path} · pas : {result.steps} · {result.model_ref.key}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export_onnx
    from .runs import RunDir

    path = Path(args.run)
    print(export_onnx(RunDir(run_id=path.name, path=path)))
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from .data import AlignedTrainingSet
    from .synthetic import write_synthetic_dataset
    from .trainer import train

    root = Path(args.runs_root)
    manifest = write_synthetic_dataset(root / "synthetic-data", 4, 2, 4, seed=0)
    config = ExperimentConfig(
        name="smoke",
        data_kind="synthetic",
        stage_channels=(8, 16, 16, 32),
        stage_blocks=(1, 1, 1, 1),
        embedding_dim=32,
        batch_size=8,
        epochs=1,
        max_steps=3,
    )
    dataset = AlignedTrainingSet.from_manifest(manifest, root / "synthetic-data")
    result = train(config, dataset, runs_root=root, manifest_counts=manifest.counts())
    print(
        f"Test de fumée SYNTHÉTIQUE terminé ({result.steps} pas, perte {result.last_loss:.3f}). "
        "Ne démontre rien sur la qualité."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m maisonaccueil.training")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--include-unassigned", action="store_true")
    p.set_defaults(func=_cmd_manifest)
    p = sub.add_parser("check")
    p.add_argument("manifest")
    p.add_argument("--disjoint", nargs=2, metavar=("SPLIT_A", "SPLIT_B"))
    p.set_defaults(func=_cmd_check)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--runs-root", default="runs")
    p.set_defaults(func=_cmd_train)
    p = sub.add_parser("export")
    p.add_argument("run")
    p.set_defaults(func=_cmd_export)
    p = sub.add_parser("smoke")
    p.add_argument("--runs-root", default="var/e/runs")
    p.set_defaults(func=_cmd_smoke)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
