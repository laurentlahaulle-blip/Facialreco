"""Rapport Markdown et export CSV des issues par passage.

Le rapport porte un AVERTISSEMENT en tête dès qu'une donnée synthétique est en jeu, et un
avertissement « test brûlé » si l'ensemble de test a servi à un réglage. Un taux n'est
jamais affiché sans ``k/n`` ; aucun score n'est présenté comme une probabilité.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from pathlib import Path

from maisonaccueil.contracts import Metric

from .annotations import AnnotationSet
from .compare import PairComparison
from .matching import OUTCOME_LABELS_FR, MatchResult
from .metrics import EngineEvaluation, Summary
from .splits import SplitReport, TestSetStatus
from .sweep import CurvePoint, ReliabilityTable, SweepPoint

SYNTHETIC_WARNING = (
    "> **AVERTISSEMENT — DONNÉES SYNTHÉTIQUES.** Ce rapport a été produit à partir de "
    "données synthétiques et de doubles de test (`Fake*`). Ses chiffres vérifient "
    "uniquement le logiciel d'évaluation ; ils **ne mesurent rien de biométrique** et ne "
    "doivent jamais être cités comme une performance de reconnaissance."
)

SCORE_NOTE = (
    "Les scores (`RawScore`) sont des scores techniques non calibrés : ce ne sont pas des "
    "probabilités ni des « % de fiabilité »."
)


def is_synthetic(ann: AnnotationSet, matches: Mapping[str, MatchResult]) -> bool:
    return ann.is_synthetic or any("synthetic" in m.data_kinds for m in matches.values())


def _esc(text: str) -> str:
    return text.replace("|", "\\|")


def _summary_rows(s: Summary) -> list[tuple[str, str]]:
    return [
        ("Passages (personnes distinctes)", f"{s.n_passages} ({s.n_persons})"),
        ("dont connus / inconnus non enregistrés", f"{s.n_known} / {s.n_unregistered}"),
        ("Bonne identification (connus)", s.correct.format()),
        ("Mauvaise identité (connus)", s.wrong_identity.format()),
        ("Connu non reconnu (connus)", s.missed.format()),
        ("Inconnu accepté à tort (inconnus)", s.false_accept.format()),
        ("Message répété (tous passages)", s.repeated.format()),
        ("Messages en trop (total)", str(s.repeated_messages_total)),
        ("Délai jusqu'à l'accueil (connus correctement accueillis)", s.delay.format()),
    ]


def render_report(
    ann: AnnotationSet,
    evaluations: Sequence[EngineEvaluation],
    *,
    split: str,
    title: str = "Rapport d'évaluation MaisonAccueil",
    generated_at: str = "",
    split_report: SplitReport | None = None,
    test_status: TestSetStatus | None = None,
    comparisons: Sequence[PairComparison] = (),
    curves: Mapping[str, Sequence[CurvePoint]] | None = None,
    sweeps: Mapping[str, Sequence[SweepPoint]] | None = None,
    reliability: Sequence[ReliabilityTable] = (),
    grace_ms: float = 1000.0,
) -> str:
    matches = {e.config_id: e.match for e in evaluations}
    lines: list[str] = [f"# {title}", ""]
    if is_synthetic(ann, matches):
        lines += [SYNTHETIC_WARNING, ""]
    if test_status is not None and test_status.burned:
        lines += [
            "> **TEST BRÛLÉ — résultats non concluants.** Cet ensemble de test a influencé un "
            "réglage ; constituer un nouvel ensemble de test indépendant avant toute "
            "conclusion. Motifs : " + " ; ".join(test_status.reasons),
            "",
        ]
    if split != "test":
        lines += [
            f"> Découpage évalué : **{split}** (réglage). Ces chiffres ont servi ou servent au "
            "réglage : ils ne constituent pas une mesure finale.",
            "",
        ]
    lines += [
        "## Protocole",
        "",
        f"- Généré le : {generated_at or 'non précisé'}",
        f"- Annotations : {', '.join(ann.sources) or '(mémoire)'} — données `{ann.data_kind}`",
        f"- Découpage évalué : `{split}` ; séances : "
        + ", ".join(f"`{s}`" for s in ann.sessions_in(split)),
        f"- Fenêtre d'attribution : [début, fin + {grace_ms:.0f} ms]",
        f"- {SCORE_NOTE}",
        "- Intervalles : Wilson 95 %, en supposant des passages indépendants (optimiste "
        "quand une même personne fait plusieurs passages ; voir le tableau par personne).",
    ]
    if test_status is not None:
        lines.append(
            f"- Ensemble de test `{test_status.set_id}` : "
            + ("**brûlé**" if test_status.burned else "non brûlé")
            + f" ({test_status.evaluations} évaluation(s) antérieure(s) enregistrée(s))"
        )
    lines.append("")
    if split_report is not None:
        lines += ["## Contrôle du découpage", ""]
        if split_report.errors:
            lines += [f"- **Erreur** : {_esc(e)}" for e in split_report.errors]
        if split_report.warnings:
            lines += [f"- Avertissement : {_esc(w)}" for w in split_report.warnings]
        if not split_report.errors and not split_report.warnings:
            lines.append("- Aucun chevauchement ni avertissement.")
        lines.append("")

    lines += ["## Résultats par configuration", ""]
    if not evaluations:
        lines += ["Aucun journal d'exécution fourni.", ""]
    for ev in evaluations:
        lines += [f"### `{_esc(ev.config_id)}`", "", "| Mesure | Valeur |", "|---|---|"]
        lines += [f"| {a} | {b} |" for a, b in _summary_rows(ev.overall)]
        lines += [
            f"| Arrivées « connu » hors de tout passage | {ev.unattributed_known} |",
            f"| Arrivées hors de tout passage (total) | {ev.unattributed_total} |",
            "",
            "#### Par personne",
            "",
            "| Personne | Passages | Bonne id. | Mauvaise id. | Non reconnu | Inconnu accepté "
            "| Message répété | Délai médian |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for person, s in ev.persons.items():
            med = f"{s.delay.median_ms:.0f} ms (n={s.delay.n})" if s.delay.n else "—"
            lines.append(
                f"| `{_esc(person)}` | {s.n_passages} | {s.correct.k}/{s.correct.n} | "
                f"{s.wrong_identity.k}/{s.wrong_identity.n} | {s.missed.k}/{s.missed.n} | "
                f"{s.false_accept.k}/{s.false_accept.n} | {s.repeated.k}/{s.repeated.n} | "
                f"{med} |"
            )
        lines.append("")

    if comparisons:
        lines += [
            "## Comparaison sur les mêmes passages",
            "",
            "Réussite = bonne identification (connus) ou rejet correct (inconnus). "
            "p : McNemar exact bilatéral, indicatif (passages supposés indépendants).",
            "",
            "| A | B | Passages | Deux réussis | A seul | B seul | Deux échoués | p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in comparisons:
            p = f"{c.mcnemar_p:.3g}" if c.mcnemar_p is not None else "—"
            lines.append(
                f"| `{_esc(c.config_a)}` | `{_esc(c.config_b)}` | {c.n} | {c.both_ok} | "
                f"{c.a_only} | {c.b_only} | {c.both_fail} | {p} |"
            )
        lines.append("")

    for name, pts in (curves or {}).items():
        lines += [
            f"## Courbe score/erreurs (règle de référence, `tune`) — `{_esc(name)}`",
            "",
            "| Seuil | Bonne id. | Mauvaise id. | Non reconnu | Inconnu accepté | Sans résultat |",
            "|---|---|---|---|---|---|",
        ]
        for pt in pts:
            lines.append(
                f"| {pt.threshold:g} | {pt.correct.k}/{pt.correct.n} | "
                f"{pt.wrong_identity.k}/{pt.wrong_identity.n} | {pt.missed.k}/{pt.missed.n} | "
                f"{pt.false_accept.k}/{pt.false_accept.n} | {pt.no_result} |"
            )
        lines.append("")
    for name, sweep in (sweeps or {}).items():
        lines += [
            f"## Balayage de la politique de décision (`tune`) — `{_esc(name)}`",
            "",
            "| Seuil | Bonne id. | Mauvaise id. | Non reconnu | Inconnu accepté | "
            "Message répété | Hors passage (connu) |",
            "|---|---|---|---|---|---|---|",
        ]
        for sp in sweep:
            s = sp.summary
            lines.append(
                f"| {sp.threshold:g} | {s.correct.k}/{s.correct.n} | "
                f"{s.wrong_identity.k}/{s.wrong_identity.n} | {s.missed.k}/{s.missed.n} | "
                f"{s.false_accept.k}/{s.false_accept.n} | {s.repeated.k}/{s.repeated.n} | "
                f"{sp.unattributed_known} |"
            )
        lines.append("")
    for tab in reliability:
        lines += [
            f"## Table de fiabilité descriptive `{tab.calibration_ref}` (non appliquée)",
            "",
            f"Modèle `{_esc(tab.model_key)}`, découpage `{tab.split}`, séances "
            + ", ".join(f"`{s}`" for s in tab.sessions)
            + ". Part des images dont le 1er candidat est la bonne personne, par tranche de "
            "score. Images d'un même passage corrélées : seule la colonne « exploitable » "
            "(assez de passages distincts) peut guider un réglage. Ce n'est pas une "
            "calibration appliquée.",
            "",
            "| Tranche | 1er candidat correct | Passages | Exploitable |",
            "|---|---|---|---|",
        ]
        for b in tab.bins:
            lines.append(
                f"| [{b.low:g} ; {b.high:g}[ | {b.top1_correct.format()} | {b.passages} | "
                f"{'oui' if b.usable else 'non'} |"
            )
        lines.append("")
    lines += [
        "## Définitions",
        "",
        "- " + "\n- ".join(f"`{o}` : {label}" for o, label in OUTCOME_LABELS_FR.items()),
        "- Message répété : plus de messages que prévu (1 si accueil attendu, 0 sinon).",
        "- Délai : premier message du bon profil − début annoté du passage.",
        "- Voir `docs/lots/G-evaluation.md` pour les règles d'attribution exactes.",
        "",
    ]
    return "\n".join(lines)


CSV_COLUMNS = (
    "config_id", "session_id", "passage_id", "expected", "profile_id", "person",
    "start_ms", "end_ms", "welcome_expected", "outcome", "known_arrivals", "unknown_arrivals",
    "messages", "repeated_messages", "delay_ms", "results", "decisions_confirmed",
    "decisions_unknown", "decisions_uncertain", "data_kinds",
)


def passages_csv(matches: Mapping[str, MatchResult]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for cfg, m in sorted(matches.items()):
        for r in m.passages:
            p = r.passage
            w.writerow([
                cfg, p.session_id, p.passage_id, str(p.expected), p.profile_id or "",
                p.person_key, p.start_ms, p.end_ms, p.welcome_expected, str(r.outcome),
                ";".join(r.known_arrivals), r.unknown_arrivals, r.messages,
                r.repeated_messages, "" if r.delay_ms is None else f"{r.delay_ms:.1f}",
                r.results, r.decisions.get("confirmed", 0), r.decisions.get("unknown", 0),
                r.decisions.get("uncertain", 0), ";".join(sorted(m.data_kinds)),
            ])
    return buf.getvalue()


def curve_csv(points: Sequence[CurvePoint], metric: Metric) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["metric", "threshold", "correct_k", "known_n", "wrong_k", "missed_k",
                "false_accept_k", "unregistered_n", "no_result"])
    for pt in points:
        w.writerow([str(metric), pt.threshold, pt.correct.k, pt.correct.n,
                    pt.wrong_identity.k, pt.missed.k, pt.false_accept.k, pt.false_accept.n,
                    pt.no_result])
    return buf.getvalue()


def write_text(path: Path | str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
