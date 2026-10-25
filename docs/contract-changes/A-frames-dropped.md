# A — Origine de `frames_dropped` et export des interfaces

**Statut** : 1 accepté (documenté dans `contracts/frames.py`) ; 2 docstring corrigée,
export refusé — les Protocols restent dans `maisonaccueil.contracts.interfaces` (import
explicite voulu). Décision coordinateur, 2026-10-25.

## 1. `CaptureState.frames_dropped`

**Besoin.** `frames_dropped` est défini comme « images abandonnées par la file bornée »,
mais la `FrameSource` ne connaît pas la `LatestQueue` du consommateur (elle ne reçoit
que `on_frame`). Le lot A contourne avec un paramètre de constructeur optionnel
`drop_counter: Callable[[], int]` (ex. `lambda: queue.dropped`).

**Changement proposé.** Documenter dans `contracts/frames.py` que la valeur provient
d'un compteur fourni par l'orchestration (sinon 0), ou bien que l'orchestration
recalcule ce champ. Aucun changement de signature nécessaire.

## 2. Export des Protocols et nom de la file

**Besoin.** `from maisonaccueil.contracts import FrameSource` échoue : les interfaces ne
sont accessibles que via `maisonaccueil.contracts.interfaces`. Par ailleurs la docstring
de `FrameSource` mentionne une `LatestFrameQueue` qui s'appelle `pipeline.LatestQueue`.

**Changement proposé.** Ajouter à `contracts/__init__.py` :

```python
from .interfaces import FrameSource  # (+ les autres Protocols)
```

et corriger la docstring (`LatestFrameQueue` → `pipeline.LatestQueue`).

**Compatibilité.** Ajout pur, aucun impact sur les lots existants.
