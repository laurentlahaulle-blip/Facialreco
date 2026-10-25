"""Adaptateur ``FaceEmbedder`` de NOTRE modèle (lot E).

``ScratchFaceEmbedder.embed(frame, face)`` :
1. exige les 5 points (``face.detection.landmarks``) fournis par le détecteur (lot D) ;
2. aligne le visage (``preprocess.align_face``, identifiant ``PREPROCESS_ID``) ;
3. calcule la représentation via un *backend* (ONNX Runtime, PyTorch, ou ``Fake*`` en test) ;
4. renvoie un ``Embedding`` float32 normalisé L2 portant le ``ModelRef`` de l'expérience.

Renvoie None si le visage est inutilisable (pas de points, trop petit, sortie invalide).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from maisonaccueil.contracts import Embedding, Frame, ModelRef, TrackedFace

from .preprocess import DEFAULT_MIN_EYE_DISTANCE_PX, PREPROCESS_ID, align_face, to_input_array
from .runs import RunDir

# (N, 3, 112, 112) float32 → (N, D) float32
EmbeddingBackend = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]


class OnnxBackend:
    """Inférence ONNX Runtime sur CPU (``onnxruntime`` importé paresseusement)."""

    def __init__(self, onnx_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dépend de l'environnement
            raise RuntimeError(
                "onnxruntime absent : installer l'extra 'recognition' dans le .venv"
            ) from exc
        self._session: Any = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def __call__(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        out = self._session.run(None, {self._input: batch})[0]
        return np.asarray(out, dtype=np.float32)


class TorchBackend:
    """Inférence PyTorch CPU depuis ``runs/<id>/weights.pt`` (torch paresseux)."""

    def __init__(self, run: RunDir) -> None:
        from .net import NormalizedEmbedding
        from .trainer import load_embedding_net, require_torch

        self._torch = require_torch()
        net, _, _ = load_embedding_net(run)
        self._module = NormalizedEmbedding(net).eval()

    def __call__(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        with self._torch.no_grad():
            out = self._module(self._torch.from_numpy(np.ascontiguousarray(batch)))
        return np.asarray(out.numpy(), dtype=np.float32)


class ScratchFaceEmbedder:
    """Implémente ``contracts.FaceEmbedder`` pour le modèle entraîné par le projet."""

    def __init__(
        self,
        model_ref: ModelRef,
        backend: EmbeddingBackend,
        min_eye_distance_px: float = DEFAULT_MIN_EYE_DISTANCE_PX,
    ) -> None:
        if model_ref.preprocess != PREPROCESS_ID:
            raise ValueError(
                f"ModelRef.preprocess={model_ref.preprocess!r} ; cet adaptateur applique "
                f"{PREPROCESS_ID!r}"
            )
        self._model = model_ref
        self._backend = backend
        self._min_eye = min_eye_distance_px

    @classmethod
    def from_run(cls, run_path: Path, backend: str = "onnx") -> ScratchFaceEmbedder:
        """Charge une expérience : ``backend`` = "onnx" (``model.onnx``) ou "torch"."""
        run = RunDir(run_id=run_path.name, path=run_path)
        ref = run.read_model_ref()
        impl: EmbeddingBackend
        if backend == "onnx":
            impl = OnnxBackend(run.onnx_path)
        elif backend == "torch":
            impl = TorchBackend(run)
        else:
            raise ValueError(f"Backend inconnu {backend!r}")
        return cls(ref, impl)

    @property
    def model(self) -> ModelRef:
        return self._model

    def embed(self, frame: Frame, face: TrackedFace) -> Embedding | None:
        landmarks = face.detection.landmarks
        if landmarks is None:
            return None
        aligned = align_face(frame.image, landmarks, self._min_eye)
        if aligned is None:
            return None
        batch = to_input_array(aligned)[None, ...]
        out = np.asarray(self._backend(batch), dtype=np.float32)
        if out.shape != (1, self._model.embedding_dim) or not np.all(np.isfinite(out)):
            return None
        vector = out[0]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return None
        return Embedding(self._model, (vector / norm).astype(np.float32))
