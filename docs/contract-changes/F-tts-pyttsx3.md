# F — Dépendance optionnelle de synthèse vocale locale (`pyttsx3`)

## Besoin

`AudioRequest.text` suppose une synthèse vocale **locale**. La bibliothèque standard n'en
fournit pas : `winsound` ne lit que des fichiers WAV. Sans synthèse, les accueils
composés à partir d'un texte (modèle « Bonjour {name} ! », message texte du profil,
message générique pour les inconnus) ne sont pas dits sous Windows.

Le lot F fournit déjà `maisonaccueil.welcome.Pyttsx3Backend`, importé paresseusement :
si le paquet est absent, `create_default_backend()` journalise un avertissement et
n'utilise que les fichiers WAV. Rien n'est cassé sans la dépendance.

## Changement proposé

Dans `pyproject.toml`, un extra optionnel (non installé par défaut) :

```toml
[project.optional-dependencies]
tts = ["pyttsx3>=2.90; sys_platform == 'win32'"]
```

puis `uv lock` par le coordinateur.

## Licences (à vérifier par le coordinateur avant ajout, à reporter dans `docs/LICENSES.md`)

| Composant | Licence (code) | Remarque |
|---|---|---|
| `pyttsx3` | MPL-2.0 (selon les métadonnées PyPI, à confirmer) | pilote les moteurs du système, hors ligne |
| `comtypes` (dépendance Windows) | MIT | |
| `pywin32` (dépendance Windows) | PSF-2.0 | |
| Voix SAPI5 | fournies avec Windows | aucune voix ni « poids » téléchargés par le projet |

Fonctionnement entièrement local (SAPI5 sous Windows). Aucun service réseau.

## Alternative sans dépendance

Appeler `System.Speech.Synthesis.SpeechSynthesizer` via PowerShell (`subprocess`).
Aucune dépendance Python, mais lancement d'un processus par message (latence ~1 s,
dépend de la politique d'exécution PowerShell). Non implémentée ; possible si
l'extra est refusé.

## Compatibilité

Aucun impact sur les contrats ni sur les autres lots. Non testé dans l'environnement
du lot F (conteneur Linux sans carte son) : seul le chemin « paquet absent » l'est.

## Statut

différé (coordinateur, 2026-10-25) — dépendance soumise à l'accord de l'utilisateur et à la
vérification des licences sur le PC Windows ; en attendant : fichiers WAV + `winsound`.
