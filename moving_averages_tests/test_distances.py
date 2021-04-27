import numpy as np
import Icp  # the old ICP calculation code
import moving_averages as siso

all_close = np.testing.assert_allclose


def test__calc_closest_point():
    # distance of a point with itself
    a = np.random.randn(3, 9)
    pts, av_dist = siso.distance.functions._calc_closest_point(a, a)
    assert (a == pts).all()
    assert av_dist == 0


class Test_conformity_to_old_functions():
    """Test that the new function are conform to the original ones."""

    def test_simple_distance(self):
        a2 = np.random.randn(3*4).reshape((-1, 3))
        a1 = np.random.randn(3*7).reshape((-1, 3))

        distance = siso.distance.calc_distance(a1, a2, 'simple')
        d_old, _, _ = Icp.Icp(a1.T, a2.T, 10, 0.1).calcSimple()

        np.testing.assert_allclose(distance, d_old)

    def test__calc_transform(self):
        a2 = np.arange(3*4).reshape((-1, 3)).T
        a1 = np.arange(3*7).reshape((-1, 3)).T
        old = Icp.Icp(a1, a2, 10, 0.1)
        old.calcTransform()

        pts, _ = siso.distance.functions._calc_closest_point(a1, a2)
        new_rot, new_tr = siso.distance.functions._calc_transform(a1, a2, pts)

        # assertions
        all_close(old.curR, new_rot)
        all_close(old.curT, new_tr)

    def test_icp(self):
        a2 = np.random.randn(3*9).reshape((-1, 3))
        a1 = np.random.randn(3*6).reshape((-1, 3))

        old = Icp.Icp(a1.T, a2.T, 10, 0.1)
        d_old, rot_old, tr_old = old.calcIcp()

        distance, rotation_M, translation_v = siso.distance.calc_distance(
            a1, a2, 'icp')

        np.testing.assert_allclose(distance, d_old)
        np.testing.assert_allclose(rotation_M, rot_old)
        np.testing.assert_allclose(translation_v, tr_old.flatten())
