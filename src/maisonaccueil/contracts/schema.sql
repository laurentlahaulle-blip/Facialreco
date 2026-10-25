-- Schéma SQLite partagé de MaisonAccueil — PROPRIÉTÉ DU COORDINATEUR.
-- Toute modification passe par une proposition (docs/contract-changes/) et une
-- incrémentation de SCHEMA_VERSION dans contracts/__init__.py.
-- Version : 1

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Personne enregistrée (volontaire). Aucun enregistrement sans consentement.
CREATE TABLE IF NOT EXISTS profiles (
    profile_id     TEXT PRIMARY KEY,              -- ProfileId "prf-..."
    display_name   TEXT NOT NULL,                 -- nom affiché
    spoken_name    TEXT,                          -- nom prononcé dans l'accueil
    welcome_text   TEXT,                          -- message personnalisé (synthèse locale)
    welcome_audio  TEXT,                          -- chemin local d'un fichier préenregistré
    consent_at     TEXT NOT NULL,                 -- ISO-8601 UTC
    consent_note   TEXT,
    active         INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

-- Séance de collecte : unité de découpage apprentissage / réglage / test.
CREATE TABLE IF NOT EXISTS collection_sessions (
    session_id   TEXT PRIMARY KEY,                -- SessionId "ses-..."
    profile_id   TEXT REFERENCES profiles(profile_id) ON DELETE CASCADE, -- NULL = inconnus
    started_at   TEXT NOT NULL,
    camera_id    TEXT,
    conditions   TEXT,                            -- éclairage, distance… (texte libre/JSON)
    split        TEXT CHECK (split IN ('train','tune','test','enroll') OR split IS NULL)
);

-- Image de référence (fichier local, jamais en base). Supprimée avec le profil.
CREATE TABLE IF NOT EXISTS reference_images (
    reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id   TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    session_id   TEXT REFERENCES collection_sessions(session_id) ON DELETE SET NULL,
    file_path    TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    captured_at  TEXT NOT NULL,
    width        INTEGER,
    height       INTEGER,
    quality_note TEXT
);

-- Représentation faciale, TOUJOURS liée à la clé exacte du modèle (ModelRef.key).
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id     TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    reference_id   INTEGER REFERENCES reference_images(reference_id) ON DELETE CASCADE,
    model_key      TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    preprocess     TEXT NOT NULL,
    metric         TEXT NOT NULL,
    dim            INTEGER NOT NULL,
    vector         BLOB NOT NULL,                 -- float32 little-endian, dim valeurs
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_key);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL                           -- JSON
);

-- Journal minimal des accueils (sans image). Purgeable.
CREATE TABLE IF NOT EXISTS arrival_log (
    event_id     TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('known','unknown')),
    profile_id   TEXT REFERENCES profiles(profile_id) ON DELETE SET NULL,
    occurred_at  TEXT NOT NULL,
    model_key    TEXT,
    policy       TEXT
);
