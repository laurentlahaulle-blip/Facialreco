# Sessions parallèles (worktrees)

Une session = une branche = un worktree = une fenêtre VS Code. Jamais deux sessions
dans le même dossier.

## Dans l'environnement cloud actuel

Les lots sont lancés comme sous-agents Claude Code isolés dans des worktrees Git
(voir le rapport du coordinateur pour la liste réelle). Ils commitent sur leur branche
`lot/<x>-<nom>` ; le coordinateur fusionne.

## Sur le PC Windows (PowerShell) — sessions manuelles

Prérequis : Git, `uv`, VS Code (commande `code`), Claude Code (commande `claude`).

```powershell
# 1. Récupérer le projet (dossier hors OneDrive)
New-Item -ItemType Directory -Force C:\DEV | Out-Null
if (Test-Path C:\DEV\MaisonAccueil) { Write-Warning "C:\DEV\MaisonAccueil existe déjà : choisir un autre nom" }
git clone https://github.com/laurentlahaulle-blip/Facialreco.git C:\DEV\MaisonAccueil
Set-Location C:\DEV\MaisonAccueil
git checkout claude/maisonaccueil-coordinator-whqt6k

# 2. Créer les worktrees (un dossier frère par lot)
$lots = @{ A="a-capture"; B="b-interface"; C="c-storage"; D="d-recognition";
           E="e-training"; F="f-decision"; G="g-evaluation" }
foreach ($k in $lots.Keys) {
  $name = $lots[$k]
  $dir  = "C:\DEV\MaisonAccueil-wt\$name"
  if (-not (Test-Path $dir)) {
    git worktree add -b "lot/$name" $dir claude/maisonaccueil-coordinator-whqt6k
  }
}
git worktree list

# 3. Ouvrir une session (exemple lot A) : nouvelle fenêtre VS Code + Claude Code
code -n C:\DEV\MaisonAccueil-wt\a-capture
#   puis, dans le terminal intégré de CETTE fenêtre :
#   uv sync --extra gui
#   claude
#   et coller le contenu de docs\prompts\A.md
```

Répéter l'étape 3 pour au plus 4 lots simultanés. Les sessions ainsi préparées
**restent à ouvrir manuellement**.

## Intégration (coordinateur uniquement)

```powershell
Set-Location C:\DEV\MaisonAccueil
git checkout claude/maisonaccueil-coordinator-whqt6k
git merge --no-ff lot/a-capture
uv run pytest
```

Ne jamais supprimer un worktree ou une branche dont le travail n'est pas intégré
(`git worktree remove` uniquement après fusion vérifiée).

## Réservation du matériel

La caméra (`camera-0`) et le GPU (`gpu-0`) sont réservés par fichier verrou dans
`%LOCALAPPDATA%\MaisonAccueil\locks` (commun à tous les worktrees), via
`maisonaccueil.pipeline.ResourceLock`.
