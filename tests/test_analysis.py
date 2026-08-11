"""
tests/test_analysis.py
======================
Testes da lógica pura (sem câmera/MediaPipe/GUI): classificação, score,
ângulo de inclinação (lean) e composição dual-camera.

Execute de qualquer lugar com:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

import config
from angle_calculator import (
    LT_SHO, RT_SHO, LT_HIP, RT_HIP, LT_EAR, RT_EAR,
    compute_all, forward_lean_angle, shoulder_tilt_angle, nose_forward_angle,
)
from posture_analyzer import (
    classify_value, score_angles, merge_dual_camera, evaluate,
)


def build_lm(points: dict, vis: float = 1.0) -> np.ndarray:
    """Cria landmarks (33, 4) com x,y definidos para `points`."""
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = vis
    for idx, (x, y) in points.items():
        lm[idx] = [x, y, 0.0, vis]
    return lm


class TestClassifyValue(unittest.TestCase):
    def test_inclusive_boundaries(self):
        bands = config.POSTURE_THRESHOLDS
        # 180° (ereto) agora cai na faixa "good" do lean (antes caía em "bad").
        self.assertEqual(classify_value(180.0, bands["lean"]), "good")
        # Limites compartilhados são inclusivos — vence a primeira banda.
        self.assertEqual(classify_value(140.0, bands["lean"]), "good")
        self.assertEqual(classify_value(15.0, bands["cva"]), "good")
        self.assertEqual(classify_value(25.0, bands["cva"]), "warning")

    def test_mid_bands(self):
        bands = config.POSTURE_THRESHOLDS
        self.assertEqual(classify_value(130.0, bands["lean"]), "warning")
        self.assertEqual(classify_value(110.0, bands["lean"]), "bad")
        self.assertEqual(classify_value(50.0, bands["lean"]), "critical")

    def test_out_of_all_bands_is_bad(self):
        bands = config.POSTURE_THRESHOLDS
        self.assertEqual(classify_value(200.0, bands["tilt"]), "bad")
        self.assertEqual(classify_value(-5.0, bands["tilt"]), "bad")


class TestScoreAngles(unittest.TestCase):
    def test_perfect_score(self):
        angles = {"tilt": 2.5, "lean": 160.0, "cva": 7.5,
                  "kyphosis": 10.0, "lordosis": 12.5}
        score, n = score_angles(angles)
        self.assertEqual(score, 100.0)
        self.assertEqual(n, 5)

    def test_unknown_angle_not_counted(self):
        angles = {"tilt": 2.5, "nome_inventado": 99.0}
        score, n = score_angles(angles)
        self.assertEqual(score, 100.0)
        self.assertEqual(n, 1)

    def test_out_of_range_zeroes(self):
        # lean em 0 → no extremo crítico → 0 pontos.
        score, n = score_angles({"lean": 0.0})
        self.assertEqual(score, 0.0)
        self.assertEqual(n, 1)

    def test_ideal_at_band_edge_is_100(self):
        # Postura PERFEITA costuma ficar na BORDA da faixa "good"
        # (ex.: lean=180°, cva=0°). Antes o score ficava preso em ~67
        # porque media a distância ao CENTRO da faixa.
        angles = {"tilt": 0.0, "lean": 180.0, "cva": 0.0,
                  "kyphosis": 0.0, "lordosis": 0.0}
        score, n = score_angles(angles)
        self.assertEqual(score, 100.0)
        self.assertEqual(n, 5)

    def test_scores_decrease_by_level(self):
        good = {"tilt": 2.5, "lean": 160.0, "cva": 7.5,
                "kyphosis": 10.0, "lordosis": 12.5}
        warning = {"tilt": 7.0, "lean": 130.0, "cva": 22.0,
                   "kyphosis": 28.0, "lordosis": 32.0}
        bad = {"tilt": 12.0, "lean": 110.0, "cva": 32.0,
               "kyphosis": 42.0, "lordosis": 47.0}
        critical = {"tilt": 20.0, "lean": 80.0, "cva": 55.0,
                    "kyphosis": 70.0, "lordosis": 70.0}
        sg, _, = score_angles(good)
        sw, _, = score_angles(warning)
        sb, _, = score_angles(bad)
        sc, _, = score_angles(critical)
        self.assertGreater(sg, sw)
        self.assertGreater(sw, sb)
        self.assertGreater(sb, sc)
        self.assertGreaterEqual(sg, 80.0)
        self.assertLessEqual(sc, 40.0)


class TestShoulderTilt(unittest.TestCase):
    """O tilt deve medir o DESVIO da linha dos ombros à horizontal (0-90°)."""

    def _lm(self, lx, ly, rx, ry):
        return build_lm({LT_SHO: (lx, ly), RT_SHO: (rx, ry)})

    def test_level_shoulders_either_orientation(self):
        # dx negativo (ombro direito à esquerda) é o NORMAL na imagem: antes
        # devolvia ~180° e derrubava o score. Nivelado deve dar ~0°.
        for dx in (-0.1, 0.1):
            ang, _ = shoulder_tilt_angle(self._lm(0.5, 0.3, 0.5 + dx, 0.3))
            self.assertAlmostEqual(ang, 0.0, delta=1.0)

    def test_angled_shoulders_magnitude(self):
        # Inclinação real ~11° (dy=0.02, dx=0.1), em qualquer orientação.
        for dx in (-0.1, 0.1):
            ang, _ = shoulder_tilt_angle(self._lm(0.5, 0.3, 0.5 + dx, 0.3 - 0.02))
            self.assertAlmostEqual(ang, 11.31, delta=1.5)


class TestNoseForward(unittest.TestCase):
    """O nose_fwd deve crescer quando o nariz avança além dos ombros."""

    def test_aligned_is_near_zero(self):
        lm = build_lm({0: (0.5, 0.25), 7: (0.5, 0.3), 8: (0.5, 0.3),
                       LT_SHO: (0.5, 0.4), RT_SHO: (0.5, 0.4)})
        ang, _ = nose_forward_angle(lm)
        self.assertLess(ang, 5.0)

    def test_forward_increases(self):
        aligned = build_lm({0: (0.5, 0.25), 7: (0.5, 0.3), 8: (0.5, 0.3),
                            LT_SHO: (0.5, 0.4), RT_SHO: (0.5, 0.4)})
        fwd = build_lm({0: (0.62, 0.25), 7: (0.5, 0.3), 8: (0.5, 0.3),
                        LT_SHO: (0.5, 0.4), RT_SHO: (0.5, 0.4)})
        a0, _ = nose_forward_angle(aligned)
        a1, _ = nose_forward_angle(fwd)
        self.assertGreater(a1, a0 + 15.0)

    def test_in_compute_all(self):
        lm = build_lm({0: (0.6, 0.25), 7: (0.5, 0.3), 8: (0.5, 0.3),
                       LT_SHO: (0.5, 0.4), RT_SHO: (0.5, 0.4)})
        self.assertIn("nose_fwd", compute_all(lm, 1.0))


class TestLeanDirection(unittest.TestCase):
    """O FLA deve punir INCLINAR P/ FRENTE e ignorar inclinar p/ trás."""

    def _lean_pair(self, dx: float):
        """lm ereto + lm inclinado por `dx` (sinal ajustado à config)."""
        sign = config.LEAN_FORWARD_X_SIGN
        base = {LT_SHO: (0.5, 0.3), RT_SHO: (0.5, 0.3)}
        erect = build_lm({**base, LT_HIP: (0.5, 0.6), RT_HIP: (0.5, 0.6)})
        fwd = build_lm({**base, LT_HIP: (0.5 + sign * dx, 0.6),
                        RT_HIP: (0.5 + sign * dx, 0.6)})
        bwd = build_lm({**base, LT_HIP: (0.5 - sign * dx, 0.6),
                        RT_HIP: (0.5 - sign * dx, 0.6)})
        return (forward_lean_angle(erect)[0],
                forward_lean_angle(fwd)[0],
                forward_lean_angle(bwd)[0])

    def test_upright_is_180(self):
        erect, _, _ = self._lean_pair(0.06)
        self.assertAlmostEqual(erect, 180.0, delta=1.0)

    def test_forward_reduces_and_backward_does_not(self):
        erect, fwd, bwd = self._lean_pair(0.06)
        self.assertLess(fwd, erect - 5.0, "inclinar p/ frente deve baixar o ângulo")
        self.assertGreaterEqual(bwd, 179.0, "inclinar p/ trás não pode ser punido")


class TestAspectCorrection(unittest.TestCase):
    def test_lean_angle_is_aspect_correct(self):
        sign = config.LEAN_FORWARD_X_SIGN
        # Em PIXELS, inclinação real de 45° (dx_px == dy_px == 48) num frame
        # 640x480 → dx_norm = 48/640, dy_norm = 48/480.
        dx, dy = 48 / 640, 48 / 480
        base = {LT_SHO: (0.5, 0.3), RT_SHO: (0.5, 0.3)}
        lm = build_lm({**base, LT_HIP: (0.5 + sign * dx, 0.3 + dy),
                       RT_HIP: (0.5 + sign * dx, 0.3 + dy)})
        correct = compute_all(lm, aspect=640 / 480)["lean"]
        distorted = compute_all(lm, aspect=1.0)["lean"]
        self.assertAlmostEqual(correct, 135.0, delta=1.0)
        self.assertGreater(abs(distorted - 135.0), abs(correct - 135.0))

    def test_low_visibility_angle_is_skipped(self):
        lm = build_lm({LT_SHO: (0.5, 0.3), RT_SHO: (0.5, 0.3),
                       LT_HIP: (0.5, 0.6), RT_HIP: (0.5, 0.6)}, vis=0.2)
        self.assertNotIn("lean", compute_all(lm, 1.0))


class TestMergeAndEvaluate(unittest.TestCase):
    def test_merge_respects_camera_roles(self):
        merged = merge_dual_camera({"tilt": 2.0}, {"lean": 150.0})
        self.assertEqual(merged, {"tilt": 2.0, "lean": 150.0})

    def test_merge_side_nose_fwd(self):
        merged = merge_dual_camera({}, {"nose_fwd": 30.0})
        self.assertEqual(merged.get("nose_fwd"), 30.0)

    def test_merge_single_camera_fallback(self):
        merged = merge_dual_camera({}, {"lean": 150.0})
        self.assertEqual(merged.get("lean"), 150.0)
        merged2 = merge_dual_camera({"lean": 150.0}, {})
        self.assertEqual(merged2.get("lean"), 150.0)

    def test_evaluate_perfect_posture(self):
        front = {"tilt": 2.5}
        side = {"cva": 7.5, "lean": 160.0, "kyphosis": 10.0, "lordosis": 12.5}
        ev = evaluate(front, side)
        self.assertEqual(ev["score"], 100.0)
        self.assertEqual(ev["worst"], "good")
        self.assertEqual(ev["cameras"], ["front", "side"])

    def test_evaluate_worst_level(self):
        front = {"tilt": 2.5}
        side = {"cva": 7.5, "lean": 90.0, "kyphosis": 10.0, "lordosis": 12.5}
        ev = evaluate(front, side)
        self.assertEqual(ev["worst"], "critical")
        self.assertEqual(ev["levels"]["lean"], "critical")


class TestSmoothing(unittest.TestCase):
    """A suavização (EMA) via evaluate(prev_angles=...) deve misturar valores."""

    def test_blends_with_previous_value(self):
        ev = evaluate({"lean": 150.0}, {}, prev_angles={"lean": 180.0}, alpha=0.5)
        # 0.5*150 + 0.5*180 = 165
        self.assertAlmostEqual(ev["angles"]["lean"]["value"], 165.0, delta=0.5)

    def test_without_prev_angles_is_pure(self):
        ev = evaluate({"lean": 150.0}, {})
        self.assertAlmostEqual(ev["angles"]["lean"]["value"], 150.0, delta=0.01)

    def test_alpha_one_is_pure(self):
        ev = evaluate({"lean": 150.0}, {}, prev_angles={"lean": 180.0}, alpha=1.0)
        self.assertAlmostEqual(ev["angles"]["lean"]["value"], 150.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
