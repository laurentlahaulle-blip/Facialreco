# F — Regroupement des accueils simultanés (« Bonjour A et B »)

## Besoin

Pour les arrivées simultanées, le lot F a choisi d'**enchaîner** les messages (voir
`docs/lots/F-decision.md`). Un regroupement en une phrase (« Bonjour Alpha et
Bravo ! ») serait plus naturel pour les messages texte, mais le lecteur ne connaît pas
le nom affichable : `AudioRequest` ne porte que `text` déjà composé et `profile_id`.

## Changement proposé (optionnel, non bloquant)

Ajouter à `AudioRequest` un champ facultatif :

```python
display_name: str | None = None  # nom à prononcer, None pour un inconnu
```

Le lecteur pourrait alors fusionner, dans une courte fenêtre (ex. 500 ms), les demandes
texte en attente qui n'ont pas de message personnalisé.

## Compatibilité

Champ facultatif avec valeur par défaut : aucun impact sur les autres lots.

## Statut

proposé
