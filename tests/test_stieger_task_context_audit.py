import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "intentflow"
    / "offline"
    / "scripts"
    / "analysis"
    / "stieger_task_context_audit.py"
)
SPEC = importlib.util.spec_from_file_location("stieger_task_context_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_tangent_identity_covariance_is_zero():
    covariance = np.eye(3, dtype=float)[None, ...]
    features = MODULE.tangent_features(covariance, np.eye(3))
    np.testing.assert_allclose(features, 0.0, atol=1e-12)


def test_one_way_icc_detects_subject_structure():
    rows = []
    for subject, center in enumerate((-5.0, 0.0, 5.0)):
        for session, noise in enumerate((-0.1, 0.0, 0.1), start=2):
            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "adapter": "prefix_ea",
                    "delta_pp": center + noise,
                }
            )
    assert MODULE.one_way_icc(rows, "prefix_ea") > 0.99
