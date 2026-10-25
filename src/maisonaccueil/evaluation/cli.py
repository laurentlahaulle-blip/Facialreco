"""Ligne de commande hors ligne : ``python -m maisonaccueil.evaluation <commande>``.

- ``check``  : contrôle du découpage d'un ou plusieurs fichiers d'annotation.
- ``report`` : évaluation de journaux d'exécution → rapport Markdown (+ CSV par passage).
- ``curve``  : courbe score/erreurs (règle de référence) sur ``tune`` → CSV.
- ``replay`` : rejeu d'une vidéo de séance avec une fabrique ``module:fonction`` qui
  renvoie un objet portant ``recognizer``, ``decision``, ``arrivals``, ``model`` et
  facultativement ``composer`` / ``policy`` (aucune implémentation importée en dur).

Sorties par défaut sous ``var/g/`` (ignoré par Git).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .annotations import DataKind, load_annotations
from .jsonl import utc_now_iso
from .protocol import evaluate
from .replay import replay_video
from .report import curve_csv, passages_csv, render_report, write_text
from .runlog import load_runlogs
from .splits import TestLedger, check_splits, sessions_from_sqlite
from .sweep import score_curve

DEFAULT_OUT = Path("var/g")


def _thresholds(spec: str) -> list[float]:
    """``"0.3:0.8:0.05"`` (début:fin:pas, fin incluse) ou ``"0.3,0.4,0.5"``."""
    if ":" in spec:
        a, b, step = (float(x) for x in spec.split(":"))
        if step <= 0:
            raise argparse.ArgumentTypeError("pas > 0 attendu")
        n = int(round((b - a) / step))
        return [round(a + i * step, 10) for i in range(n + 1)]
    return [float(x) for x in spec.split(",") if x.strip()]


def _load_factory(spec: str) -> Any:
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--factory attend 'module:fonction'")
    return getattr(importlib.import_module(mod_name), attr)


def cmd_check(args: argparse.Namespace) -> int:
    ann = load_annotations(*args.annotations)
    extra = sessions_from_sqlite(args.db) if args.db else []
    rep = check_splits(ann, extra)
    if ann.is_synthetic:
        print("AVERTISSEMENT : annotations synthétiques (tests logiciels uniquement).")
    for split, c in rep.counts.items():
        print(f"{split}: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    for e in rep.errors:
        print(f"ERREUR : {e}")
    for w in rep.warnings:
        print(f"Avertissement : {w}")
    return 0 if rep.ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    ann = load_annotations(*args.annotations)
    logs = load_runlogs(args.runs)
    ledger = TestLedger(args.ledger) if args.ledger else None
    extra = sessions_from_sqlite(args.db) if args.db else []
    run = evaluate(ann, logs, split=args.split, ledger=ledger, allow_burned=args.allow_burned,
                   grace_ms=args.grace_ms, extra_sessions=extra)
    text = render_report(ann, run.evaluations, split=args.split, generated_at=utc_now_iso(),
                         split_report=run.split_report, test_status=run.test_status,
                         comparisons=run.comparisons, grace_ms=args.grace_ms)
    write_text(args.out, text)
    print(f"Rapport écrit : {args.out}")
    if args.csv:
        write_text(args.csv, passages_csv({e.config_id: e.match for e in run.evaluations}))
        print(f"CSV écrit : {args.csv}")
    return 0


def cmd_curve(args: argparse.Namespace) -> int:
    ann = load_annotations(*args.annotations)
    logs = load_runlogs(args.runs)
    ledger = TestLedger(args.ledger) if args.ledger else None
    pts = score_curve(ann, logs, _thresholds(args.thresholds), grace_ms=args.grace_ms,
                      ledger=ledger)
    write_text(args.out, curve_csv(pts, logs[0].header.model.metric))
    print(f"Courbe écrite : {args.out}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    setup = _load_factory(args.factory)()
    log, stats = replay_video(
        args.video,
        session_id=args.session,
        config_id=args.config_id,
        model=setup.model,
        recognizer=setup.recognizer,
        decision=setup.decision,
        arrivals=setup.arrivals,
        composer=getattr(setup, "composer", None),
        policy=getattr(setup, "policy", ""),
        out_path=args.out,
        data_kind=DataKind(args.data_kind),
        fps=args.fps,
    )
    print(f"Journal écrit : {args.out} ({stats.frames} images, {stats.results} résultats, "
          f"{stats.arrivals} arrivées, {stats.welcomes} accueils)")
    del log
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m maisonaccueil.evaluation",
                                description="Évaluation hors ligne MaisonAccueil (lot G)")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="contrôle du découpage")
    c.add_argument("annotations", nargs="+", type=Path)
    c.add_argument("--db", type=Path, help="base SQLite (collection_sessions, lecture seule)")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("report", help="rapport Markdown")
    r.add_argument("--annotations", nargs="+", type=Path, required=True)
    r.add_argument("--runs", nargs="+", type=Path, required=True)
    r.add_argument("--split", choices=("tune", "test"), default="test")
    r.add_argument("--ledger", type=Path, default=DEFAULT_OUT / "ledger.jsonl")
    r.add_argument("--allow-burned", action="store_true")
    r.add_argument("--db", type=Path)
    r.add_argument("--grace-ms", type=float, default=1000.0)
    r.add_argument("--out", type=Path, default=DEFAULT_OUT / "report.md")
    r.add_argument("--csv", type=Path)
    r.set_defaults(func=cmd_report)

    k = sub.add_parser("curve", help="courbe score/erreurs sur tune")
    k.add_argument("--annotations", nargs="+", type=Path, required=True)
    k.add_argument("--runs", nargs="+", type=Path, required=True)
    k.add_argument("--thresholds", required=True)
    k.add_argument("--ledger", type=Path, default=DEFAULT_OUT / "ledger.jsonl")
    k.add_argument("--grace-ms", type=float, default=1000.0)
    k.add_argument("--out", type=Path, default=DEFAULT_OUT / "curve.csv")
    k.set_defaults(func=cmd_curve)

    v = sub.add_parser("replay", help="rejeu d'une vidéo de séance")
    v.add_argument("--video", type=Path, required=True)
    v.add_argument("--session", required=True)
    v.add_argument("--config-id", required=True)
    v.add_argument("--factory", required=True, help="module:fonction sans argument")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--data-kind", choices=("real", "synthetic"), default="real")
    v.add_argument("--fps", type=float)
    v.set_defaults(func=cmd_replay)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
