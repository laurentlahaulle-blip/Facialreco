"""Export ONNX du réseau de représentation d'une expérience.

Entrée ``input`` : float32 ``(N, 3, 112, 112)`` (voir ``preprocess``) ; sortie
``embedding`` : float32 ``(N, embedding_dim)`` normalisée L2. Le fichier ``model.onnx``
et ``onnx.json`` (empreinte, opset, clé du ``ModelRef``) restent dans ``runs/<id>/``.
"""

from __future__ import annotations

from pathlib import Path

from .preprocess import INPUT_SIZE, PREPROCESS_ID
from .runs import RunDir, file_sha256, model_ref_to_dict
from .trainer import load_embedding_net, require_torch

DEFAULT_OPSET = 17


def export_onnx(run: RunDir, opset: int = DEFAULT_OPSET, path: Path | None = None) -> Path:
    torch = require_torch()
    from .net import NormalizedEmbedding

    net, config, ref = load_embedding_net(run)
    wrapper = NormalizedEmbedding(net).eval()
    target = path or run.onnx_path
    dummy = torch.zeros(1, 3, config.input_size, config.input_size, dtype=torch.float32)
    kwargs = {
        "input_names": ["input"],
        "output_names": ["embedding"],
        "dynamic_axes": {"input": {0: "batch"}, "embedding": {0: "batch"}},
        "opset_version": opset,
    }
    with torch.no_grad():
        try:
            # Exporteur TorchScript : ne dépend pas d'onnxscript.
            torch.onnx.export(wrapper, (dummy,), str(target), dynamo=False, **kwargs)
        except TypeError:  # versions de torch sans l'argument ``dynamo``
            torch.onnx.export(wrapper, (dummy,), str(target), **kwargs)
    run.write_json(
        "onnx.json",
        {
            "file": target.name,
            "onnx_sha256": file_sha256(target),
            "opset": opset,
            "input": {
                "name": "input",
                "shape": ["N", 3, INPUT_SIZE, INPUT_SIZE],
                "preprocess": PREPROCESS_ID,
            },
            "output": {"name": "embedding", "normalized": "l2"},
            "model_ref": model_ref_to_dict(ref),
        },
    )
    return target
