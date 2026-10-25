# Lot C — Stockage SQLite

Paquet : `maisonaccueil.storage`. Schéma : `contracts/schema.sql` v1 (non modifié).

## API publique

| Classe | Rôle |
|---|---|
| `Database(path, media_root=None, busy_timeout_ms=5000)` | ouverture, schéma, WAL, `foreign_keys=ON`, version ; `conn` (connexion du thread), `transaction()`, `close()`, `schema_version`, `media_root` |
| `SchemaVersionError` | base de version supérieure (ou inférieure, aucune migration) |
| `ProfileRepository` | `create(display_name, *, consent_at, ...)`, `get`, `find`, `list(include_inactive=False)`, `update(**champs)`, `renew_consent`, `deactivate`, `activate`, `set_active`, `delete(profile_id, purge_arrivals=True) -> DeletionReport` |
| `ReferenceRepository` | séances : `create_session`, `get_session`, `list_sessions`, `set_session_split` ; images : `add_image_bytes`, `import_file`, `get`, `list_for_profile`, `list_for_session`, `verify`, `delete` |
| `SqliteEmbeddingStore(db, include_inactive=False)` | implémente `contracts.interfaces.EmbeddingStore` : `add_embedding`, `load_embeddings`, `delete_embeddings_for_model` ; en plus `represented_reference_ids(model)` (proposition D-represented-references : profils actifs ET inactifs, indépendant de `include_inactive`), `count`, `model_keys`, `delete_embeddings_for_profile` |
| `SettingsRepository` | `get(key, default)`, `set`, `set_many`, `delete`, `all` (JSON strict) |
| `ArrivalLogRepository` | `record(event, model_key, policy)` (idempotent), `list_recent`, `count`, `purge_older_than(cutoff)`, `purge_all` |

## Choix

- **Thread-safety** : une connexion SQLite par thread (`threading.local`) ; écritures en
  `BEGIN IMMEDIATE` ; WAL + `busy_timeout` sérialisent les écrivains. `:memory:` refusé.
- **Horodatages** : écrits en ISO-8601 UTC à la microseconde (`+00:00`) ; datetimes naïfs refusés.
- **Consentement** : `consent_at` obligatoire et timezone-aware à la création.
- **Images** : écrites sous `media_root/<profile_id>/ref-<uuid>.<ext>` (écriture atomique),
  sha256 en base. Un fichier n'est jamais effacé s'il est hors de `media_root`.
- **Suppression complète** d'un profil : représentations, références, séances, journal
  d'arrivées (par défaut) puis fichiers ; `DeletionReport` donne les compteurs et les
  fichiers ignorés.
- **Représentations** : float32 little-endian ; filtrage par `model_key` puis revérification
  de toutes les colonnes du `ModelRef` et de la taille du BLOB (`CorruptEmbeddingError`) ;
  vecteur de dimension/dtype incohérent ou NaN/Inf refusé ; référence d'un autre profil
  refusée. `load_embeddings` exclut les profils désactivés par défaut.

## Tests

`uv run pytest tests/storage` — bases dans `tmp_path`, octets d'image synthétiques.

## Limites / pistes (non bloquantes, aucune proposition de schéma déposée)

- Pas de migration de schéma (une seule version existe).
- Index `profile_id` absents sur `embeddings` / `reference_images` (volumes familiaux : sans effet).
- `arrival_log` : le schéma garde la ligne avec `profile_id` NULL à la suppression ;
  le lot C purge ces lignes par défaut (`purge_arrivals=True`).
