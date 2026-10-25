"""Journal d'exécution (JSON Lines) : résultats, décisions, arrivées, accueils horodatés.

Format ``maisonaccueil.runlog`` version 1 — un fichier par (séance, configuration).
Tous les ``t_ms`` sont en **temps de séance** : ``(monotonic_ns - origin_monotonic_ns) / 1e6``,
le même repère que ``start_ms``/``end_ms`` des annotations de passages.

``run`` (première ligne) ::

    {"record": "run", "format": "maisonaccueil.runlog", "version": 1,
     "run_id": "run-...", "session_id": "ses-...", "config_id": "<moteur + politique + seuils>",
     "model": {"name", "version", "embedding_dim", "metric", "preprocess", "origin"},
     "policy": "temporal-v1", "data_kind": "real" | "synthetic",
     "source": "replay:<nom de fichier>" | "live", "origin_monotonic_ns": 0,
     "messages_from": "welcome" | "arrival", "started_at": "...Z",
     "contracts_version": "1.0.0", "note": ""}

``frame`` — une image traitée (facultatif, sert au rejeu fidèle des ``tick``) ::

    {"record": "frame", "t_ms": 1234.5, "frame_id": 37, "processing_ms": 12.3}

``result`` — un ``RecognitionResult`` (candidats triés, score technique NON calibré) ::

    {"record": "result", "t_ms": 1234.5, "frame_id": 37, "track_id": "trk-...",
     "metric": "cosine", "candidates": [{"profile_id": "prf-...", "score": 0.71, "refs": 5}],
     "quality_flags": ["blurry"]}

``decision`` ::

    {"record": "decision", "t_ms", "track_id", "status": "confirmed|unknown|uncertain",
     "profile_id": null | "prf-...", "best_score": null | 0.71, "observations": 3,
     "window_ms": 800, "policy": "temporal-v1", "reason": ""}

``track_lost`` : ``{"record": "track_lost", "t_ms", "track_id"}``

``arrival`` ::

    {"record": "arrival", "t_ms", "event_id", "kind": "known|unknown", "track_id",
     "profile_id": null | "prf-...", "wall_time": "...Z"}

``welcome`` — demande d'accueil acceptée par le lecteur (le texte n'est pas journalisé) ::

    {"record": "welcome", "t_ms", "request_id", "arrival_event_id", "profile_id",
     "dedupe_key", "mode": "audio_path" | "text"}

Aucune image, aucune représentation faciale n'est écrite dans le journal.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from maisonaccueil.contracts import (
    CONTRACTS_VERSION,
    ArrivalEvent,
    AudioRequest,
    Candidate,
    Decision,
    FrameId,
    Metric,
    ModelRef,
    ProfileId,
    QualityFlag,
    RawScore,
    RecognitionResult,
    SessionId,
    TrackId,
)

from .annotations import DataKind
from .jsonl import FormatError, dumps, read_jsonl, require, utc_now_iso

RUNLOG_FORMAT = "maisonaccueil.runlog"
RUNLOG_VERSION = 1
EVENT_RECORDS = ("frame", "result", "decision", "track_lost", "arrival", "welcome")


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def model_to_json(m: ModelRef) -> dict[str, Any]:
    return {
        "name": m.name,
        "version": m.version,
        "embedding_dim": m.embedding_dim,
        "metric": str(m.metric),
        "preprocess": m.preprocess,
        "origin": m.origin,
    }


def model_from_json(obj: Mapping[str, Any], where: str) -> ModelRef:
    try:
        return ModelRef(
            name=str(obj["name"]),
            version=str(obj["version"]),
            embedding_dim=int(obj["embedding_dim"]),
            metric=Metric(obj["metric"]),
            preprocess=str(obj["preprocess"]),
            origin=str(obj["origin"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise FormatError(f"{where}: champ 'model' invalide ({exc})") from None


@dataclass(frozen=True, slots=True)
class RunHeader:
    session_id: SessionId
    config_id: str
    model: ModelRef
    data_kind: DataKind
    run_id: str = field(default_factory=new_run_id)
    policy: str = ""
    source: str = "live"
    origin_monotonic_ns: int = 0
    messages_from: str = "welcome"  # "welcome" ou "arrival" (pas de compositeur)
    started_at: str = field(default_factory=utc_now_iso)
    contracts_version: str = CONTRACTS_VERSION
    note: str = ""

    def __post_init__(self) -> None:
        if self.messages_from not in ("welcome", "arrival"):
            raise ValueError("messages_from doit valoir 'welcome' ou 'arrival'")
        if not self.config_id:
            raise ValueError("config_id obligatoire (identifie moteur + politique + seuils)")

    def to_json(self) -> dict[str, Any]:
        return {
            "record": "run",
            "format": RUNLOG_FORMAT,
            "version": RUNLOG_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "config_id": self.config_id,
            "model": model_to_json(self.model),
            "policy": self.policy,
            "data_kind": str(self.data_kind),
            "source": self.source,
            "origin_monotonic_ns": self.origin_monotonic_ns,
            "messages_from": self.messages_from,
            "started_at": self.started_at,
            "contracts_version": self.contracts_version,
            "note": self.note,
        }


class RunLogWriter:
    """Écrit les enregistrements d'une exécution (fichier ou mémoire)."""

    def __init__(self, header: RunHeader, path: Path | str | None = None) -> None:
        self.header = header
        self.records: list[dict[str, Any]] = [header.to_json()]
        self._fh: IO[str] | None = None
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("w", encoding="utf-8", newline="\n")
            self._fh.write(dumps(self.records[0]) + "\n")

    def t_ms(self, monotonic_ns: int) -> float:
        return (monotonic_ns - self.header.origin_monotonic_ns) / 1e6

    def _emit(self, rec: dict[str, Any]) -> None:
        self.records.append(rec)
        if self._fh is not None:
            self._fh.write(dumps(rec) + "\n")

    def frame(self, frame_id: int, monotonic_ns: int, processing_ms: float) -> None:
        self._emit({"record": "frame", "t_ms": self.t_ms(monotonic_ns), "frame_id": frame_id,
                    "processing_ms": round(processing_ms, 3)})

    def result(self, r: RecognitionResult) -> None:
        self._emit(
            {
                "record": "result",
                "t_ms": self.t_ms(r.monotonic_ns),
                "frame_id": int(r.frame_id),
                "track_id": r.track_id,
                "metric": str(r.model.metric),
                "candidates": [
                    {"profile_id": c.profile_id, "score": c.score.value,
                     "refs": c.references_compared}
                    for c in r.candidates
                ],
                "quality_flags": sorted(str(f) for f in r.quality_flags),
            }
        )

    def decision(self, d: Decision) -> None:
        self._emit(
            {
                "record": "decision",
                "t_ms": self.t_ms(d.monotonic_ns),
                "track_id": d.track_id,
                "status": str(d.status),
                "profile_id": d.profile_id,
                "best_score": d.best_score.value if d.best_score is not None else None,
                "observations": d.observations,
                "window_ms": d.window_ms,
                "policy": d.policy,
                "reason": d.reason,
            }
        )

    def track_lost(self, track_id: TrackId, monotonic_ns: int) -> None:
        self._emit({"record": "track_lost", "t_ms": self.t_ms(monotonic_ns),
                    "track_id": track_id})

    def arrival(self, e: ArrivalEvent) -> None:
        self._emit(
            {
                "record": "arrival",
                "t_ms": self.t_ms(e.monotonic_ns),
                "event_id": e.event_id,
                "kind": str(e.kind),
                "track_id": e.track_id,
                "profile_id": e.profile_id,
                "wall_time": e.wall_time.isoformat().replace("+00:00", "Z"),
            }
        )

    def welcome(self, a: AudioRequest) -> None:
        self._emit(
            {
                "record": "welcome",
                "t_ms": self.t_ms(a.monotonic_ns),
                "request_id": a.request_id,
                "arrival_event_id": a.arrival_event_id,
                "profile_id": a.profile_id,
                "dedupe_key": a.dedupe_key,
                "mode": "audio_path" if a.audio_path is not None else "text",
            }
        )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def to_runlog(self) -> RunLog:
        return parse_runlog(self.records, where="<mémoire>")

    def __enter__(self) -> RunLogWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class RunLog:
    header: RunHeader
    events: tuple[dict[str, Any], ...]  # triés par t_ms (ordre d'écriture conservé à égalité)
    source_path: str = ""

    def of(self, *records: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["record"] in records]

    def recognition_results(self) -> list[RecognitionResult]:
        """Reconstruit les ``RecognitionResult`` (pour rejouer la décision)."""
        out: list[RecognitionResult] = []
        m = self.header.model
        origin = self.header.origin_monotonic_ns
        for e in self.of("result"):
            metric = Metric(e["metric"])
            cands = tuple(
                Candidate(ProfileId(c["profile_id"]), RawScore(float(c["score"]), metric),
                          int(c.get("refs", 0)))
                for c in e["candidates"]
            )
            out.append(
                RecognitionResult(
                    frame_id=FrameId(int(e["frame_id"])),
                    track_id=TrackId(e["track_id"]),
                    monotonic_ns=origin + round(float(e["t_ms"]) * 1e6),
                    model=m,
                    candidates=cands,
                    quality_flags=frozenset(QualityFlag(f) for f in e.get("quality_flags", [])),
                )
            )
        return out


def parse_runlog(records: Iterable[Mapping[str, Any]], where: str) -> RunLog:
    it = iter(records)
    try:
        head = dict(next(it))
    except StopIteration:
        raise FormatError(f"{where}: journal vide") from None
    if head.get("record") != "run":
        raise FormatError(f"{where}: la première ligne doit être 'run'")
    if head.get("format") != RUNLOG_FORMAT or head.get("version") != RUNLOG_VERSION:
        raise FormatError(f"{where}: format attendu {RUNLOG_FORMAT} v{RUNLOG_VERSION}")
    try:
        header = RunHeader(
            session_id=SessionId(require(head, "session_id", str, where)),
            config_id=require(head, "config_id", str, where),
            model=model_from_json(require(head, "model", dict, where), where),
            data_kind=DataKind(require(head, "data_kind", str, where)),
            run_id=require(head, "run_id", str, where),
            policy=str(head.get("policy") or ""),
            source=str(head.get("source") or ""),
            origin_monotonic_ns=int(head.get("origin_monotonic_ns") or 0),
            messages_from=str(head.get("messages_from") or "welcome"),
            started_at=str(head.get("started_at") or ""),
            contracts_version=str(head.get("contracts_version") or ""),
            note=str(head.get("note") or ""),
        )
    except ValueError as exc:
        if isinstance(exc, FormatError):
            raise
        raise FormatError(f"{where}: en-tête invalide ({exc})") from None
    events: list[dict[str, Any]] = []
    for i, rec in enumerate(it, start=2):
        rtype = rec.get("record")
        if rtype not in EVENT_RECORDS:
            raise FormatError(f"{where}#{i}: enregistrement inconnu {rtype!r}")
        require(rec, "t_ms", (int, float), f"{where}#{i}")
        events.append(dict(rec))
    events.sort(key=lambda e: float(e["t_ms"]))  # tri stable
    return RunLog(header, tuple(events), where)


def load_runlog(path: Path | str) -> RunLog:
    return parse_runlog((obj for _, obj in read_jsonl(path)), where=str(path))


def load_runlogs(paths: Sequence[Path | str]) -> list[RunLog]:
    return [load_runlog(p) for p in paths]
