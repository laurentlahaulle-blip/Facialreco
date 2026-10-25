"""``SqliteEmbeddingStore`` : implémentation SQLite de ``contracts.interfaces.EmbeddingStore``.

- Vecteurs float32 little-endian en BLOB (``dim`` valeurs).
- Filtrage STRICT par ``ModelRef.key`` ; chaque ligne relue est revérifiée (nom, version,
  prétraitement, métrique, dimension, taille du BLOB) : une incohérence lève
  ``CorruptEmbeddingError`` au lieu de renvoyer une représentation douteuse.
- Par défaut, ``load_embeddings`` ne renvoie que les profils ACTIFS (un profil désactivé
  ne doit pas être reconnu) ; ``include_inactive=True`` pour les outils hors ligne.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import numpy as np

from maisonaccueil.contracts import Embedding, ModelRef, ProfileId

from .database import Database, utc_now_iso

_DTYPE = np.dtype("<f4")


class CorruptEmbeddingError(RuntimeError):
    """Ligne stockée incohérente avec la clé de modèle demandée."""


def encode_vector(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype=_DTYPE).tobytes()


def decode_vector(blob: bytes, dim: int) -> np.ndarray:
    if len(blob) != dim * _DTYPE.itemsize:
        raise CorruptEmbeddingError(f"BLOB de {len(blob)} octets pour dim={dim}")
    return np.frombuffer(blob, dtype=_DTYPE).astype(np.float32)  # copie native, modifiable


class SqliteEmbeddingStore:
    def __init__(self, db: Database, *, include_inactive: bool = False) -> None:
        self._db = db
        self._include_inactive = include_inactive

    def add_embedding(
        self, profile_id: ProfileId, reference_id: int | None, embedding: Embedding
    ) -> int:
        model = embedding.model
        vec = embedding.vector
        if vec.ndim != 1 or vec.shape[0] != model.embedding_dim:
            raise ValueError(
                f"Dimension {vec.shape} incohérente avec {model.key} (dim={model.embedding_dim})"
            )
        if vec.dtype != np.float32:
            raise ValueError("Le vecteur doit être float32")
        if not np.all(np.isfinite(vec)):
            raise ValueError("Vecteur contenant NaN/Inf refusé")
        with self._db.transaction() as c:
            if reference_id is not None:
                row = c.execute(
                    "SELECT profile_id FROM reference_images WHERE reference_id = ?",
                    (reference_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"reference_id inconnu : {reference_id}")
                if row[0] != profile_id:
                    raise ValueError("La référence appartient à un autre profil")
            cur = c.execute(
                "INSERT INTO embeddings(profile_id, reference_id, model_key, model_name,"
                " model_version, preprocess, metric, dim, vector, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    profile_id, reference_id, model.key, model.name, model.version,
                    model.preprocess, str(model.metric), model.embedding_dim,
                    sqlite3.Binary(encode_vector(vec)), utc_now_iso(),
                ),
            )
            return int(cur.lastrowid or 0)

    def load_embeddings(self, model: ModelRef) -> Iterator[tuple[ProfileId, Embedding]]:
        """Représentations du modèle ``model`` (même ``key``), ordre stable.

        La liste est matérialisée avant d'être renvoyée (pas de curseur ouvert laissé à
        l'appelant)."""
        return iter(self._load(model))

    def _load(self, model: ModelRef) -> list[tuple[ProfileId, Embedding]]:
        sql = (
            "SELECT e.embedding_id, e.profile_id, e.model_name, e.model_version,"
            " e.preprocess, e.metric, e.dim, e.vector"
            " FROM embeddings e JOIN profiles p ON p.profile_id = e.profile_id"
            " WHERE e.model_key = ?"
        )
        if not self._include_inactive:
            sql += " AND p.active = 1"
        sql += " ORDER BY e.embedding_id"
        out: list[tuple[ProfileId, Embedding]] = []
        for r in self._db.conn.execute(sql, (model.key,)):
            stored = (r["model_name"], r["model_version"], r["preprocess"], r["metric"],
                      int(r["dim"]))
            expected = (model.name, model.version, model.preprocess, str(model.metric),
                        model.embedding_dim)
            if stored != expected:
                raise CorruptEmbeddingError(
                    f"embedding_id={r['embedding_id']} : colonnes {stored} ≠ {expected}"
                )
            vec = decode_vector(bytes(r["vector"]), model.embedding_dim)
            out.append((ProfileId(r["profile_id"]), Embedding(model=model, vector=vec)))
        return out

    def count(self, model: ModelRef | None = None) -> int:
        if model is None:
            return int(self._db.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
        return int(
            self._db.conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_key = ?", (model.key,)
            ).fetchone()[0]
        )

    def model_keys(self) -> dict[str, int]:
        """Clés de modèle présentes et nombre de représentations par clé."""
        rows = self._db.conn.execute(
            "SELECT model_key, COUNT(*) FROM embeddings GROUP BY model_key ORDER BY model_key"
        )
        return {r[0]: int(r[1]) for r in rows}

    def delete_embeddings_for_model(self, model: ModelRef) -> int:
        with self._db.transaction() as c:
            return int(
                c.execute("DELETE FROM embeddings WHERE model_key = ?", (model.key,)).rowcount
            )

    def delete_embeddings_for_profile(
        self, profile_id: ProfileId, model: ModelRef | None = None
    ) -> int:
        with self._db.transaction() as c:
            if model is None:
                cur = c.execute("DELETE FROM embeddings WHERE profile_id = ?", (profile_id,))
            else:
                cur = c.execute(
                    "DELETE FROM embeddings WHERE profile_id = ? AND model_key = ?",
                    (profile_id, model.key),
                )
            return int(cur.rowcount)
