"""Тесты этапа 1 — сферические гармоники, мультипольный базис, вращения.

Проверяются: табличные значения и фазы, ортонормировность (символьное
интегрирование), лестничные рекурсии производных, собственный смысл оператора
Бельтрами, поле :math:`\\ell=1` как диполь, произвольный порядок ``l``
(подстановка вместо захардкоженных формул), STF-тензоры, унитарность матриц
Вигнера и согласованность пересчёта моментов с фазой аксиального вращения.
"""

import itertools

import numpy as np
import pytest
import sympy as sp

from mag_nav.symbolic.core.conventions import I, MU0_OVER_4PI, OMEGA, TIME, demod_phase
from mag_nav.symbolic.core.coords import (
    PHI,
    R,
    THETA,
    X,
    Y as YC,
    Z,
    eliminate_r,
    spherical_frame_matrix,
    spherical_to_cartesian,
    unit_r,
)
from mag_nav.symbolic.core.tensor import Tensor
from mag_nav.symbolic.harmonics.multipole import (
    CARTESIAN_AXES,
    cartesian_stf_multipole,
    multipole_field,
    multipole_field_cartesian,
    multipole_potential,
    stf_moment_symbols,
    stf_multipole_field,
    stf_multipole_potential,
    stf_radial_tensor,
)
from mag_nav.symbolic.harmonics.spherical import (
    Y,
    Y_conjugate_relation,
    Y_explicit,
    Y_general,
    check_orthogonality,
    dY_dphi,
    dY_dtheta,
    dY_dtheta_down_ladder,
    grad_Y,
    laplace_beltrami,
    norm_integral,
    orthogonality_integral,
    physical_gradient_spherical,
    surface_gradient,
)
from mag_nav.symbolic.harmonics.wigner import (
    axial_phase,
    clebsch_gordan,
    euler_zyz_from_matrix,
    m_order,
    rotate_moment,
    rotation_matrix_zyz,
    wigner_D,
    wigner_D_element,
    wigner_D_matrix,
    wigner_d_small_matrix,
)

ELL = sp.Symbol("ell", integer=True, nonnegative=True)
SQRT_PI = sp.sqrt(sp.pi)


def trigsimplify(expr):
    """Надёжное упрощение тригонометрических тождеств (plain simplify слабее)."""
    return sp.simplify(sp.expand_trig(sp.sympify(expr)))


# ---------------------------------------------------------------------------
# Табличные значения и фазовая конвенция
# ---------------------------------------------------------------------------


def test_harmonics_table_values_and_condon_shortley_phase():
    """Явные значения Y_lm совпадают с таблицей, фаза Кондона — Шортли на месте."""
    table = {
        (0, 0): 1 / (2 * SQRT_PI),
        (1, 0): sp.sqrt(3 / (4 * sp.pi)) * sp.cos(THETA),
        (1, 1): -sp.sqrt(3 / (8 * sp.pi)) * sp.sin(THETA) * sp.exp(I * PHI),
        (1, -1): sp.sqrt(3 / (8 * sp.pi)) * sp.sin(THETA) * sp.exp(-I * PHI),
        (2, 0): sp.sqrt(5 / (16 * sp.pi)) * (3 * sp.cos(THETA) ** 2 - 1),
        (2, 2): sp.sqrt(15 / (32 * sp.pi)) * sp.sin(THETA) ** 2 * sp.exp(2 * I * PHI),
    }
    for (l, m), expected in table.items():
        assert trigsimplify(Y(l, m) - expected) == 0, (l, m)
    # знак именно Condon-Shortley: Y_1^1 < 0 при sin(theta)>0
    assert Y(1, 1).args_as_coefficients_dict is not None
    assert sp.re(sp.N(Y(1, 1).subs({THETA: 1.0, PHI: 0.0}))) < 0


def test_Y_general_matches_explicit_and_symbolic():
    """Замкнутая формула через assoc_legendre: совпадает с явной и живёт при символьном l."""
    for l in range(0, 4):
        for m in range(-l, l + 1):
            assert trigsimplify(Y_general(l, m) - Y_explicit(l, m)) == 0, (l, m)
    # символьный l остаётся символьным
    expr = Y_general(ELL, 1)
    assert ELL in expr.free_symbols
    assert sp.simplify(expr.subs(ELL, 2) - Y_explicit(2, 1)) == 0


def test_conjugation_relation():
    """Связь сопряжения: ``conj(Y_lm) = (-1)^m Y_{l,-m}``."""
    for l in (0, 1, 2, 3):
        for m in range(-l, l + 1):
            assert trigsimplify(sp.conjugate(Y(l, m)) - Y_conjugate_relation(l, m)) == 0, (l, m)


# ---------------------------------------------------------------------------
# Ортогональность и нормировка (символьное интегрирование)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l1,m1,l2,m2", [
    (1, 0, 1, 1),
    (1, 1, 1, -1),
    (2, 1, 2, -1),
    (2, 0, 0, 0),
    (2, 1, 3, 1),
    (3, 2, 2, 2),
])
def test_orthogonality_of_distinct_pairs(l1, m1, l2, m2):
    """``int Y*_{l1m1} Y_{l2m2} dOmega = 0`` для разных пар."""
    assert orthogonality_integral(l1, m1, l2, m2) == 0
    assert check_orthogonality(l1, m1, l2, m2)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_normalization(l):
    """``int |Y_lm|^2 dOmega = 1`` для всех m при l = 0..3."""
    for m in range(-l, l + 1):
        assert norm_integral(l, m) == 1, (l, m)


def test_orthogonality_identical_pair_gives_one():
    """Ортогональность «в себе»: ``int |Y_11|^2 = 1``, ``check_orthogonality`` даёт True."""
    assert check_orthogonality(1, 1, 1, 1) is True
    assert orthogonality_integral(1, 1, 1, 1) == 1


# ---------------------------------------------------------------------------
# Производные: рекурсии и градиент
# ---------------------------------------------------------------------------


def test_phi_derivative_is_im_Y_for_symbolic_l():
    """``dY/dphi = I*m*Y`` — и для символьного ``l`` тоже."""
    assert trigsimplify(dY_dphi(ELL, 1) - I * 1 * Y(ELL, 1)) == 0
    assert trigsimplify(dY_dphi(3, -2) - I * (-2) * Y(3, -2)) == 0


@pytest.mark.parametrize("l,m", [(1, 0), (1, 1), (2, 1), (2, 2), (3, -2), (4, 2)])
def test_theta_ladder_recurrences_agree(l, m):
    """Две лестничные рекурсии (вверх/вниз) дают одну и ту же производную."""
    assert trigsimplify(dY_dtheta(l, m) - dY_dtheta_down_ladder(l, m)) == 0, (l, m)


def test_theta_derivative_matches_direct_differentiation():
    """Рекурсия согласована с прямой дифференциацией явного выражения."""
    for (l, m) in [(2, 1), (3, 0), (3, 3)]:
        direct = sp.diff(Y_explicit(l, m), THETA)
        assert trigsimplify(dY_dtheta(l, m) - direct) == 0, (l, m)


def test_gradient_of_inverse_radius_is_minus_rhat_over_r_squared():
    """``grad(1/r) = -r_hat/r^2``: и в сферическом кадре, и в декартовом."""
    g_sph = physical_gradient_spherical(1 / R)
    assert trigsimplify(g_sph[0] + 1 / R ** 2) == 0
    assert trigsimplify(g_sph[1]) == 0 and trigsimplify(g_sph[2]) == 0

    f_cart = Tensor.scalar(1 / R, name="1/r")
    g_cart = f_cart.applyfunc(eliminate_r).grad(CARTESIAN_AXES, name="grad(1/r)")
    reference = Tensor([-X / (X ** 2 + YC ** 2 + Z ** 2) ** sp.Rational(3, 2),
                        -YC / (X ** 2 + YC ** 2 + Z ** 2) ** sp.Rational(3, 2),
                        -Z / (X ** 2 + YC ** 2 + Z ** 2) ** sp.Rational(3, 2)], name="ref")
    assert g_cart.equals(reference)


def test_grad_Y_components_and_cartesian_agreement():
    """``grad_Y``: нулевая радиальная компонента, согласование кадров для l <= 2."""
    for (l, m) in [(1, 0), (1, 1), (2, -1)]:
        g = grad_Y(l, m, frame="spherical")
        assert g.rank == 1 and trigsimplify(g[0]) == 0
        # угловая часть = поверхностный градиент / r
        s = surface_gradient(l, m)
        assert trigsimplify(g[1] - s[1] / R) == 0
        assert trigsimplify(g[2] - s[2] / R) == 0
        # прямая проверка: компоненты в декартовом базисе = dY/dx_i
        g_cart = grad_Y(l, m, frame="cartesian")
        Y_cart = sp.simplify(spherical_to_cartesian(Y_explicit(l, m), keep_r=True))
        direct = [sp.simplify(sp.diff(Y_cart, axis)) for axis in CARTESIAN_AXES]
        for i in range(3):
            assert trigsimplify(g_cart[i] - direct[i]) == 0, (l, m, i)


def test_laplace_beltrami_eigenvalue():
    """``Delta_Omega Y_lm = -l(l+1) Y_lm`` (символьно, включая символьный l)."""
    for l in (0, 1, 2, 3):
        for m in range(-l, l + 1):
            residual = laplace_beltrami(Y(l, m)).as_scalar() + l * (l + 1) * Y(l, m)
            assert trigsimplify(residual) == 0, (l, m)
    # и для символьного порядка: оператор даёт -l(l+1) Я
    sym_res = trigsimplify(laplace_beltrami(Y(ELL, 1)).as_scalar() + ELL * (ELL + 1) * Y(ELL, 1))
    assert sym_res == 0, sym_res


# ---------------------------------------------------------------------------
# Мультипольный базис
# ---------------------------------------------------------------------------


def test_multipole_potential_is_Y_over_r_to_lplus1():
    """Потенциал: ``Y_lm / r^(l+1)``, символьный ``l`` сохраняется."""
    assert trigsimplify(multipole_potential(2, 1) - Y(2, 1) / R ** 3) == 0
    phi = multipole_potential(ELL, 1)
    assert ELL in phi.free_symbols
    assert sp.simplify(phi.subs(ELL, 3) - multipole_potential(3, 1)) == 0


def test_multipole_field_symbolic_l_matches_substitution():
    """Поле при символьном ``l`` переходит в поле конкретного порядка подстановкой."""
    g_sym = multipole_field(ELL, 1, frame="spherical")
    for l in (1, 2, 3):
        g_num = g_sym.subs(ELL, l).simplify()
        assert g_num.equals(multipole_field(l, 1, frame="spherical")), l


def test_multipole_field_equals_negative_gradient_of_potential():
    """Определение ``g = -grad Phi`` проверено независимым дифференцированием (l <= 3)."""
    for (l, m) in [(1, -1), (2, 0), (3, 2)]:
        phi_cart = Tensor.scalar(
            sp.simplify(spherical_to_cartesian(multipole_potential(l, m), keep_r=True)), name="phi"
        )
        grad = phi_cart.grad(CARTESIAN_AXES)
        g = multipole_field(l, m, frame="cartesian")
        for i in range(3):
            assert trigsimplify(g[i] + grad[i]) == 0, (l, m, i)


def test_multipole_field_cartesian_matches_direct_differentiation():
    """``multipole_field_cartesian`` --- компоненты через x, y, z; сверка с прямым grad."""
    for (l, m) in [(1, 0), (1, 1), (2, -1), (3, 2)]:
        g = multipole_field_cartesian(l, m)
        assert THETA not in g.free_symbols and PHI not in g.free_symbols
        phi = sp.simplify(spherical_to_cartesian(multipole_potential(l, m), keep_r=True))
        for i, axis in enumerate(CARTESIAN_AXES):
            assert trigsimplify(g[i] + sp.diff(phi, axis)) == 0, (l, m, i)
    # символьный l для декартовой замкнутой формы запрещён (и это объясняется)
    with pytest.raises(ValueError):
        multipole_field_cartesian(ELL, 1)


def test_dipole_field_matches_known_formula_m_zero():
    """l=1, m=0: поле совпадает с диполем вдоль z (известная формула)."""
    m0 = sp.Symbol("m0", positive=True)
    g = multipole_field_cartesian(1, 0)
    # B = (mu0/4pi) * m0 * sqrt(4pi/3) * g,  m = m0 * e_z
    pref = MU0_OVER_4PI * m0 * sp.sqrt(4 * sp.pi / 3)
    B = Tensor([sp.simplify(pref * c) for c in g], name="B")
    r2 = X ** 2 + YC ** 2 + Z ** 2
    m_dot_rhat = m0 * Z / sp.sqrt(r2)
    ref = Tensor([
        sp.simplify(pref * sp.sqrt(3 / (4 * sp.pi)) * (3 * Z * v / r2 ** sp.Rational(5, 2))
                   - (m0 * (pref / m0) * 0))
        for v in (0, 0, 0)
    ], name="unused")
    # явный эталон: (mu0/4pi)/r^3 * [3 (m.rhat) rhat - m] с m = m0 e_z
    rhat = [X / sp.sqrt(r2), YC / sp.sqrt(r2), Z / sp.sqrt(r2)]
    mvec = [sp.Integer(0), sp.Integer(0), m0]
    reference = []
    for i in range(3):
        expr = MU0_OVER_4PI / r2 ** sp.Rational(3, 2) * (
            3 * m_dot_rhat * rhat[i] - mvec[i]
        )
        reference.append(sp.simplify(expr))
    assert B.equals(Tensor(reference, name="reference"))


def test_dipole_field_matches_known_formula_m_minus1():
    """l=1, m=-1: вращающийся диполь ``m = m0(e_x - I e_y)`` — поле из явной формулы."""
    m0 = sp.Symbol("m0", positive=True)
    g = multipole_field_cartesian(1, -1)
    pref = MU0_OVER_4PI * m0 * sp.sqrt(8 * sp.pi / 3)
    B = Tensor([sp.simplify(pref * c) for c in g], name="B")
    r2 = X ** 2 + YC ** 2 + Z ** 2
    rhat = [X / sp.sqrt(r2), YC / sp.sqrt(r2), Z / sp.sqrt(r2)]
    mvec = [m0, -I * m0, sp.Integer(0)]
    m_dot_rhat = sp.simplify(sum(mvec[i] * rhat[i] for i in range(3)))
    reference = [sp.simplify(MU0_OVER_4PI / r2 ** sp.Rational(3, 2) *
                             (3 * m_dot_rhat * rhat[i] - mvec[i])) for i in range(3)]
    assert B.equals(Tensor(reference, name="reference"))


def test_cartesian_stf_multipoles_explicit_and_traceless():
    """СТФ-базис: известные формулы l=1,2; симметрия и нулевой след при l<=3."""
    M1 = cartesian_stf_multipole(1)
    assert M1.equals(Tensor([-X / R ** 3, -YC / R ** 3, -Z / R ** 3], name="ref"))

    M2 = cartesian_stf_multipole(2)
    delta = Tensor.delta(3)
    ref = (Tensor([X, YC, Z], name="r").outer(Tensor([X, YC, Z], name="r")) * 3
           - delta * R ** 2) / R ** 5
    assert M2.equals(ref.simplify())

    for l in (1, 2, 3):
        M = cartesian_stf_multipole(l)
        assert M.rank == l and M.shape == (3,) * l
        assert M.is_symmetric(0, 1)
        assert M.is_traceless(0, 1)  # след = лапласиан 1/r = 0 вне источника


def test_stf_radial_tensor_double_factorial_identity():
    """``partial_K(1/r) = (-1)^l (2l-1)!! r_hat_<K> / r^(l+1)`` для l = 1..3."""
    for l in (1, 2, 3):
        M = cartesian_stf_multipole(l)
        n_stf = stf_radial_tensor(l)
        double_fact = sp.factorial2(2 * l - 1)
        ref = n_stf * ((-1) ** l * double_fact / R ** (l + 1))
        assert M.equals(ref.simplify()), l


def test_stf_multipole_potential_dipole_and_quadrupole():
    """Декартов мультипольный потенциал: диполь и квадруполь в известном виде."""
    q = Tensor([sp.Symbol("qx"), sp.Symbol("qy"), sp.Symbol("qz")], name="q")
    phi = stf_multipole_potential(q)
    ref = sp.simplify((q[0] * X + q[1] * YC + q[2] * Z) / R ** 3)
    assert sp.simplify(restore_check(phi - ref)) == 0

    # квадруполь: симметричный бесследовый Q_ij -> (3/2) Q_ij x_i x_j / r^5
    Qs = sp.symbols("Qxx Qxy Qxz Qyy Qyz Qzz", real=True)
    Q = Tensor([[Qs[0], Qs[1], Qs[2]], [Qs[1], Qs[3], Qs[4]], [Qs[2], Qs[4], Qs[5]]], name="Q")
    phi2 = stf_multipole_potential(Q)
    contraction = sum(Q[i, j] * X * X if i == j == 0 else 0 for i in range(3) for j in range(3))
    quad = sp.Integer(0)
    for i in range(3):
        for j in range(3):
            quad += Q[i, j] * CARTESIAN_AXES[i] * CARTESIAN_AXES[j]
    ref2 = sp.simplify(sp.Rational(3, 2) * quad / R ** 5)
    assert sp.simplify(restore_check(phi2.as_scalar() - ref2)) == 0


def restore_check(expr):
    """Приводит ``x^2+y^2+z^2`` обратно к ``r^2`` для удобства сверки."""
    from mag_nav.symbolic.core.coords import restore_r

    return restore_r(expr)


def test_stf_multipole_field_is_harmonic():
    """Поле STF-мультиполя вне источника удовлетворяет уравнению Лапласа."""
    for l in (1, 2, 3):
        moment = stf_moment_symbols(l, prefix=f"Q{l}")
        g = stf_multipole_field(moment)
        assert g.rank == 1
        divergence = sp.simplify(sum(sp.diff(eliminate_r(g[i]), CARTESIAN_AXES[i]) for i in range(3)))
        assert divergence == 0, l


# ---------------------------------------------------------------------------
# Матрицы Вигнера и повороты
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l", [0, 1, 2])
def test_wigner_D_unitary_and_identity_at_zero(l):
    """``D^dagger D = I``; при нулевых углах ``D = I``."""
    a, b, g = sp.symbols("alpha beta gamma", real=True)
    D = wigner_D_matrix(l, a, b, g)
    prod = (D.adjoint() * D).applyfunc(lambda e: sp.simplify(sp.expand_trig(e)))
    assert prod == sp.eye(2 * l + 1)
    assert wigner_D_matrix(l, 0, 0, 0) == sp.eye(2 * l + 1)
    # Tensor-обёртка сохраняет форму и ранг
    T = wigner_D(l, a, b, g)
    assert T.rank == 2 and T.shape == (2 * l + 1, 2 * l + 1)
    assert T.indices == ("m_p", "m")


def test_wigner_D_axis_rotation_is_diagonal_phase():
    """Поворот вокруг z: ``D_{m'm}(alpha,0,0) = exp(-I m alpha) delta_{m'm}``."""
    alpha = sp.Symbol("alpha", real=True)
    for l in (1, 2):
        D = wigner_D_matrix(l, alpha, 0, 0)
        ms = m_order(l)
        for i, mi in enumerate(ms):
            for j, mj in enumerate(ms):
                expected = sp.exp(-I * mi * alpha) if i == j else sp.Integer(0)
                assert sp.simplify(D[i, j] - expected) == 0, (l, mi, mj)


def test_wigner_D_element_matches_matrix_and_small_d(l):
    """Отдельный элемент согласован с матрицей; малая матрица при нулевых фазах."""
    l = 2
    a, b, g = sp.symbols("alpha beta gamma", real=True)
    for mp, m in itertools.product(range(-l, l + 1), repeat=2):
        elem = wigner_D_element(l, mp, m, a, b, g)
        assert sp.simplify(elem - wigner_D_matrix(l, a, b, g)[m_order(l).index(mp),
                                                             m_order(l).index(m)]) == 0
    # d(beta) = D(0, beta, 0)
    d_small = wigner_d_small_matrix(l, b)
    D0 = wigner_D_matrix(l, 0, b, 0)
    assert (d_small - D0).applyfunc(sp.simplify) == sp.zeros(2 * l + 1, 2 * l + 1)


def test_wigner_D_composition_for_consecutive_rotations():
    """Композиция поворотов: ``D(Rz(a1)) D(Rz(a2)) = D(Rz(a1+a2))`` (аксиальный случай)."""
    a1, a2 = sp.symbols("alpha_1 alpha_2", real=True)
    l = 2
    D1 = wigner_D_matrix(l, a1, 0, 0)
    D2 = wigner_D_matrix(l, a2, 0, 0)
    D12 = wigner_D_matrix(l, a1 + a2, 0, 0)
    assert (D1 * D2 - D12).applyfunc(lambda e: sp.simplify(sp.expand_trig(e))) == sp.eye(2 * l + 1)


def test_wigner_D_rotation_act_on_harmonics_numeric():
    """Закон вращения гармоник: ``Y_lm(R^-1 r) = sum_{m'} D*_{m'm} Y_lm'(r)`` (численно)."""
    rng = np.random.default_rng(20261108)
    for l in (1, 2):
        angles = rng.uniform(-1.5, 1.5, size=3)
        a, b, g = [sp.nsimplify(float(v)) for v in angles]
        Rmat = rotation_matrix_zyz(a, b, g)
        Rinv = Rmat.T
        D = wigner_D_matrix(l, a, b, g)
        ms = m_order(l)
        # точка на сфере: theta, phi -> декартовы -> поворот -> обратно в углы
        theta0, phi0 = 1.1, 2.3
        vec = sp.Matrix([sp.sin(THETA) * sp.cos(PHI), sp.sin(THETA) * sp.sin(PHI), sp.cos(THETA)])
        vec_r = (Rinv * vec).subs({THETA: theta0, PHI: phi0})
        th1 = float(sp.acos(vec_r[2]))
        ph1 = float(sp.atan2(vec_r[1], vec_r[0]))
        f_rot = sp.lambdify((THETA, PHI), Y_explicit(l, 1), "numpy")
        lhs = complex(f_rot(th1, ph1))
        rhs = 0j
        for i, mp in enumerate(ms):
            coeff = complex(sp.N(sp.conjugate(D[i, ms.index(1)])))
            val = complex(sp.lambdify((THETA, PHI), Y_explicit(l, mp), "numpy")(theta0, phi0))
            rhs += coeff * val
        assert abs(lhs - rhs) < 1e-12, (l, lhs, rhs)


def test_axial_phase_matches_conventions_and_D():
    """Фаза аксиального вращения согласована с ``conventions.demod_phase`` и ``D``."""
    for m in (-2, -1, 0, 1, 2):
        assert sp.simplify(axial_phase(2, m, OMEGA, TIME) - demod_phase(m, OMEGA, TIME)) == 0
    # и как элемент D для поворота на omega*t вокруг z
    l = 2
    D = wigner_D_matrix(l, OMEGA * TIME, 0, 0)
    ms = m_order(l)
    for m in ms:
        assert sp.simplify(D[ms.index(m), ms.index(m)] - axial_phase(l, m, OMEGA, TIME)) == 0


def test_rotate_moment_matches_axial_phase():
    """Поворот моментов вокруг z даёт ровно фазу ``exp(-I m omega t)``."""
    q = {(1, m): sp.Symbol(f"q1m{m}".replace("-", "m")) for m in (-1, 0, 1)}
    rotated = rotate_moment(q, euler=(OMEGA * TIME, 0, 0))
    for m in (-1, 0, 1):
        key = (1, m)
        assert sp.simplify(rotated[key] - axial_phase(1, m, OMEGA, TIME) * q[key]) == 0, m


def test_rotate_moment_preserves_norm_for_real_rotation_numeric():
    """Для вещественного поворота сумма |q_lm|^2 не меняется (унитарность D)."""
    l = 2
    values = {(l, m): complex(1, m) for m in range(-l, l + 1)}
    a, b, g = 0.4, -0.7, 1.2
    rotated = rotate_moment(values, euler=(a, b, g))
    before = sum(abs(v) ** 2 for v in values.values())
    after = sum(abs(complex(sp.N(v))) ** 2 for v in rotated.values())
    assert abs(before - after) < 1e-12


def test_euler_zyz_roundtrip():
    """``euler_zyz_from_matrix(rotation_matrix_zyz(...))`` воспроизводит углы."""
    a, b, g = sp.symbols("alpha beta gamma", real=True)
    R = rotation_matrix_zyz(a, b, g)
    a2, b2, g2 = euler_zyz_from_matrix(R)
    # сверка численно (atan2 даёт главные значения)
    subs = {a: 0.6, b: 0.9, g: -1.1}
    R_num = np.array(R.subs(subs).evalf(), dtype=float)
    R_back = np.array(rotation_matrix_zyz(a2.subs(subs), b2.subs(subs), g2.subs(subs)).evalf(),
                      dtype=float)
    assert np.max(np.abs(R_num - R_back)) < 1e-12


def test_clebsch_gordan_values_and_sum_rule():
    """Клебш—Гордан: табличные значения и правило суммы по конечным каналам."""
    # <1 1; 1 -1 | 0 0> = 1/sqrt(3), <1 0; 1 0 | 0 0> = -1/sqrt(3)
    assert sp.simplify(clebsch_gordan(1, 1, 1, -1, 0, 0) - 1 / sp.sqrt(3)) == 0
    assert sp.simplify(clebsch_gordan(1, 0, 1, 0, 0, 0) + 1 / sp.sqrt(3)) == 0
    # набор по всем L при фиксированных m1,m2: сумма квадратов = 1
    block = clebsch_gordan(1, 1, 1, -1)
    total = sum(c ** 2 for c in block.values())
    assert sp.simplify(total - 1) == 0
    # 1 (x) 1 содержит L = 0, 1, 2 (след, антисимметричная, STF-части)
    assert set(k[0] for k in clebsch_gordan(1, 1, 1, 0)) == {1, 2}


def test_spherical_frame_matrix_used_by_rotation_is_orthogonal():
    """Матрица кадра, используемая ``rotate_to_cartesian_frame``, ортогональна."""
    M = spherical_frame_matrix()
    assert (M.T * M - sp.eye(3)).applyfunc(lambda e: sp.simplify(sp.expand_trig(e))) == sp.zeros(3, 3)
