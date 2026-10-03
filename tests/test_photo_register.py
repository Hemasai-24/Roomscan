import numpy as np

from roomscan.photo_register import rigid2d_ransac


def _rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_recovers_rotation_and_shift_despite_outliers():
    rng = np.random.default_rng(0)
    b = rng.uniform(-3, 3, (200, 2))
    R, t = _rot(37), np.array([4.0, -1.5])
    a = b @ R.T + t + rng.normal(0, 0.01, b.shape)
    a[:60] = rng.uniform(-5, 5, (60, 2))                  # 30 % wrong matches
    R2, t2, inl = rigid2d_ransac(a, b)
    np.testing.assert_allclose(R2, R, atol=0.01)
    np.testing.assert_allclose(t2, t, atol=0.03)
    assert inl.sum() >= 130 and inl[:60].sum() <= 2      # a random wrong match can land on its true spot


def test_deterministic():
    rng = np.random.default_rng(1)
    b = rng.uniform(-3, 3, (50, 2))
    a = b @ _rot(10).T + 1.0
    r1 = rigid2d_ransac(a, b)
    r2 = rigid2d_ransac(a, b)
    np.testing.assert_array_equal(r1[0], r2[0])


def test_too_few_points_returns_none():
    assert rigid2d_ransac(np.zeros((2, 2)), np.zeros((2, 2))) is None
