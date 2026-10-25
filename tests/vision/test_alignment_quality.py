import cv2
import numpy as np
import pytest

from maisonaccueil.contracts import BoundingBox, Detection, FrameId, Landmarks5, QualityFlag
from maisonaccueil.vision import (
    ARCFACE_TEMPLATE_112,
    PREPROCESS_ID,
    AlignmentError,
    QualityConfig,
    align_face,
    assess_quality,
    landmarks_from_bbox,
)
from maisonaccueil.vision.alignment import similarity_matrix


def _transformed_landmarks(scale: float, angle_deg: float, tx: float, ty: float) -> Landmarks5:
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * scale
    pts = ARCFACE_TEMPLATE_112.astype(np.float64) @ rot.T + np.array([tx, ty])
    return Landmarks5(tuple((float(x), float(y)) for x, y in pts))


def test_preprocess_id_is_versioned():
    assert PREPROCESS_ID.endswith("-v1")


def test_alignment_maps_landmarks_onto_template():
    lm = _transformed_landmarks(scale=2.5, angle_deg=20, tx=120, ty=40)
    m = similarity_matrix(lm)
    src = np.asarray(lm.points, np.float64)
    mapped = src @ m[:, :2].T + m[:, 2]
    assert np.allclose(mapped, ARCFACE_TEMPLATE_112, atol=1e-2)


def test_align_face_moves_synthetic_markers_to_template():
    img = np.zeros((480, 640, 3), np.uint8)
    lm = _transformed_landmarks(scale=3.0, angle_deg=-15, tx=200, ty=60)
    for x, y in lm.points:
        cv2.circle(img, (round(x), round(y)), 6, (255, 255, 255), -1)
    before = img.copy()
    aligned = align_face(img, lm)
    assert aligned.shape == (112, 112, 3) and aligned.dtype == np.uint8
    assert np.array_equal(img, before)  # l'image source n'est pas modifiée
    for tx, ty in ARCFACE_TEMPLATE_112:
        assert aligned[round(ty), round(tx)].max() > 200


def test_alignment_rejects_non_finite_points():
    bad = Landmarks5(((float("nan"), 0.0),) + ((1.0, 1.0),) * 4)
    with pytest.raises(AlignmentError):
        align_face(np.zeros((50, 50, 3), np.uint8), bad)


def test_landmarks_from_bbox_inside_box():
    lm = landmarks_from_bbox(10, 20, 100, 100)
    assert all(10 <= x <= 110 and 20 <= y <= 120 for x, y in lm.points)


def _det(x, y, w, h, landmarks=None) -> Detection:
    return Detection(FrameId(0), BoundingBox(x, y, w, h), 0.9, "test", landmarks)


def _textured(h=240, w=320, value=150):
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // 4) + (xx // 4)) % 2) * 60 - 30
    img = np.clip(value + checker, 0, 255).astype(np.uint8)
    return np.dstack([img] * 3)


def test_quality_good_face_has_no_flags():
    lm = landmarks_from_bbox(50, 50, 100, 100)
    flags = assess_quality(_textured(), _det(50, 50, 100, 100, lm))
    assert flags == frozenset()


def test_quality_small_face():
    assert QualityFlag.FACE_TOO_SMALL in assess_quality(_textured(), _det(10, 10, 20, 20))


def test_quality_blurry_and_low_light():
    flat_bright = np.full((240, 320, 3), 180, np.uint8)
    assert QualityFlag.BLURRY in assess_quality(flat_bright, _det(50, 50, 100, 100))
    dark = _textured(value=15)
    assert QualityFlag.LOW_LIGHT in assess_quality(dark, _det(50, 50, 100, 100))


def test_quality_not_frontal_from_landmarks():
    pts = list(landmarks_from_bbox(50, 50, 100, 100).points)
    pts[2] = (pts[1][0] + 5, pts[2][1])  # nez décalé au-delà de l'œil droit
    flags = assess_quality(_textured(), _det(50, 50, 100, 100, Landmarks5(tuple(pts))))
    assert QualityFlag.NOT_FRONTAL in flags


def test_quality_config_is_respected():
    cfg = QualityConfig(min_face_px=10)
    assert QualityFlag.FACE_TOO_SMALL not in assess_quality(_textured(), _det(10, 10, 20, 20), cfg)
