"""Тесты этапа 0 — тензорный объект, координаты, соглашения.

Каждое тождество проверяется символьно: ``assert sp.simplify(lhs - rhs) == 0``
(для тензоров — через :meth:`Tensor.equals`, которая делает то же покомпонентно).
"""

import itertools

import pytest
import sympy as sp

from mag_nav.symbolic.core.conventions import (
    MU0_OVER_4PI,
    OMEGA,
    TIME,
    complex_moment,
    complex_pair,
    demod_phase,
    describe_conventions,
    is_concrete_integer,
    real_signal,
    time_phase,
    validate_lm,
)
from mag_nav.symbolic.core.coords import (
    PHI,
    R,
    THETA,
    X,
    Y,
    Z,
    basis_tensor,
    cartesian_to_spherical,
    cyclic_basis_vector,
    cyclic_to_cartesian_matrix,
    eliminate_r,
    levi_civita,
    metric,
    position_tensor,
    spherical_frame_matrix,
    spherical_to_cartesian,
    unit_phi,
    unit_r,
    unit_theta,
)
from mag_nav.symbolic.core.tensor import Tensor, stf_coefficients

a, b, c, d = sp.symbols("a b c d", real=True)
p, q, s, w = sp.symbols("p q s w")


# ---------------------------------------------------------------------------
# Хранение и доступ
# ---------------------------------------------------------------------------


def test_tensor_stores_and_returns_components():
    """Tensor корректно хранит и возвращает компоненты (ранг 1, 2, 3, 0)."""
    v = Tensor([a, b, c], name="v")
    assert v.rank == 1 and v.shape == (3,)
    assert v[0] == a and v[2] == c
    assert v.components() == [a, b, c]

    m = Tensor([[a, b], [c, d]], name="M")
    assert m.rank == 2 and m.shape == (2, 2)
    assert m[1, 0] == c
    assert m.as_matrix() == sp.Matrix([[a, b], [c, d]])

    t3 = Tensor([[[a, b], [c, d]], [[d, c], [b, a]]], name="T3")
    assert t3.rank == 3 and t3.shape == (2, 2, 2)
    assert t3[1, 0, 1] == c
    assert len(t3) == 8

    sc = Tensor.scalar(a * p, name="phi")
    assert sc.rank == 0 and sc.shape == ()
    assert sc.as_scalar() == a * p
    assert sc[()] == a * p

    from_mat = Tensor.from_matrix(sp.Matrix([p, q]), name="v2")
    assert from_mat.rank == 1 and from_mat[1] == q

    assert v.free_symbols == {a, b, c}


def test_tensor_metadata_and_repr():
    """Метаданные (имя, симметрия, индексы) и представление."""
    t = Tensor([[a, b], [c, d]], name="Q", symmetry="symmetric", indices=("i", "j"))
    assert t.name == "Q" and t.symmetry == "symmetric" and t.indices == ("i", "j")
    assert "rank=2" in repr(t) and "symmetric" in repr(t)
    with pytest.raises(ValueError):
        Tensor([[a, b], [c, d]], indices=("i",))  # длина indices != ранг


# ---------------------------------------------------------------------------
# Свёртка, внешнее произведение
# ---------------------------------------------------------------------------


def test_contract_with_delta_is_identity():
    """Свёртка с дельтой: ``delta_ij T_jk = T_ik`` и ``delta_ij T_ij = Tr T``."""
    T = Tensor([[a, b], [c, d]], name="T")
    delta = Tensor.delta(2)
    # оси произведения: (d0, d1, T0, T1); свёртка d1 (ось 1) с T0 (ось 2)
    res = delta.outer(T).contract(1, 2)
    assert res.rank == 2
    assert res.equals(T)

    tr = T.trace()
    assert tr.rank == 0 and sp.simplify(tr.as_scalar() - (a + d)) == 0

    # свёртка ранга 3: delta_ij T_jkl = T_ikl
    T3 = Tensor([[[a, b], [c, d]], [[d, a], [b, c]]], name="T3")
    res3 = Tensor.delta(2).outer(T3).contract(1, 2)
    assert res3.rank == 3 and res3.equals(T3)


def test_outer_rank_and_shape():
    """Внешнее произведение: ранг складывается, форма конкатенируется."""
    v = Tensor([a, b], name="v")
    w = Tensor([p, q, s], name="w")
    vw = v.outer(w)
    assert vw.rank == 2 and vw.shape == (2, 3)
    assert vw[1, 2] == b * s
    vvw = v.outer(v).outer(w)
    assert vvw.rank == 3 and vvw.shape == (2, 2, 3)
    # смешивание со скаляром
    sc = Tensor.scalar(c, name="c")
    assert v.outer(sc).rank == 1 and v.outer(sc).equals(v * c)


def test_inner_matches_matmul():
    """``inner`` и ``matmul`` согласованы (матричное произведение)."""
    A = Tensor([[a, b], [c, d]], name="A")
    B = Tensor([[p, q], [s, w]], name="B")
    AB = A.matmul(B)
    ref = (A.as_matrix() * B.as_matrix())
    assert AB.equals(Tensor.from_matrix(ref, name="ref"))

    v = Tensor([a, b], name="v")
    dot = A.matmul(v)
    assert dot.rank == 1 and dot.equals(Tensor.from_matrix(A.as_matrix() * v.as_vector()))

    scalar = v.matmul(w_vec())
    assert scalar.rank == 0


def w_vec():
    return Tensor([p, q], name="w")


# ---------------------------------------------------------------------------
# Симметризация / антисимметризация
# ---------------------------------------------------------------------------


def test_sym_plus_antisym_reconstructs_tensor():
    """Симметричная + антисимметричная части = исходный тензор (ранг 2)."""
    T = Tensor([[a, b], [c, d]], name="T")
    sym = T.symmetrize()
    asym = T.antisymmetrize()
    assert sym.is_symmetric() and asym.is_antisymmetric()
    assert (sym + asym).equals(T)
    assert sym.symmetry == "symmetric" and asym.symmetry == "antisymmetric"


def test_symmetrize_idempotent_and_partial_axes():
    """Симметризация идемпотентна; частичная — только по выбранным осям."""
    T = Tensor([[a, b], [c, d]], name="T")
    assert T.symmetrize().symmetrize().equals(T.symmetrize())

    # ранг 3: симметризуем только оси (1, 2)
    M = Tensor([[[a, b], [c, d]], [[p, q], [s, w]]], name="M").permute((0, 1, 2))
    S = M.symmetrize((1, 2))
    assert S.is_symmetric(1, 2)
    assert not S.is_symmetric(0, 1)

    # полная симметризация ранга 3 даёт симметрию по всем парам
    F = M.symmetrize()
    for (i, j) in itertools.combinations(range(3), 2):
        assert F.is_symmetric(i, j)


def test_antisymmetrize_of_outer_is_wedge():
    """Антисимметризованное внешнее произведение двух векторов = «клинь»."""
    u = Tensor([a, b, c], name="u")
    v = Tensor([p, q, s], name="v")
    wedge = u.outer(v).antisymmetrize()
    assert wedge.is_antisymmetric()
    # явный эталон: 2*W_ik = u_i v_k - u_k v_i
    ui = [a, b, c]
    vk = [p, q, s]
    ref = [[ui[i] * vk[j] - ui[j] * vk[i] for j in range(3)] for i in range(3)]
    assert (wedge * 2).equals(Tensor(ref, name="ref"))


# ---------------------------------------------------------------------------
# След и STF
# ---------------------------------------------------------------------------


def test_traceless_rank2_has_zero_trace():
    """Снятие следа: ``Tr(Q) = 0`` для результата (ранг 2)."""
    T = Tensor([[a, b], [c, d]], name="T")
    Q = T.traceless()
    assert Q.symmetry == "traceless"
    assert sp.simplify(Q.trace().as_scalar()) == 0
    assert Q.is_traceless()
    # явная формула: T - delta*Tr/2 (размерность 2)
    ref = T - Tensor.delta(2) * (a + d) / 2
    assert Q.equals(ref)


def test_traceless_rank3_zero_pair_trace():
    """Снятие следа по паре осей тензора ранга 3 обнуляет именно этот след."""
    T = Tensor([[[a, b], [c, d]], [[p, q], [s, w]]], name="T")
    Q = T.traceless(0, 1)
    assert Q.rank == 3
    tr = Q.trace(0, 1)
    assert all(sp.simplify(x) == 0 for x in tr)


def test_stf_projector_coefficients():
    """Коэффициенты STF: закрытая формула совпадает с точным линейным выводом."""
    from mag_nav.symbolic.core.tensor import stf_coefficients_closed_form, stf_coefficients_derived

    # известные значения (выведены модулем, не подсмотрены из справочника)
    assert stf_coefficients(1, 3) == (sp.Integer(1),)
    assert stf_coefficients(2, 3) == (sp.Integer(1), sp.Rational(-1, 3))
    assert stf_coefficients(3, 3) == (sp.Integer(1), sp.Rational(-3, 5))
    # независимая проверка: линейно-алгебраический вывод совпадает при rank <= 4
    for rank in (2, 3, 4):
        for dim in (2, 3, 4):
            assert stf_coefficients_closed_form(rank, dim) == stf_coefficients_derived(rank, dim), (
                f"rank={rank}, dim={dim}"
            )
    # знакочередование, c_0 == 1, длина --- при произвольном ранге
    for rank in (5, 6, 7, 8):
        coef = stf_coefficients(rank, 3)
        assert coef[0] == 1 and len(coef) == rank // 2 + 1
        assert all(sp.sign(c) == (-1) ** s for s, c in enumerate(coef[1:], start=1))


def test_stf_is_symmetric_and_all_traces_vanish():
    """STF: симметричность (соседние транспозиции) + обнуление следа (одна пара).

    Для полностью симметричного тензора все парные следы равны с точностью до
    перестановки осей, поэтому довольно проверить пару (0, 1); соседние
    транспозиции порождают всю S_k, значит довольно и их.
    """
    for rank, size in ((2, 2), (3, 3), (4, 3)):
        comps = sp.symbols(f"t0:{size ** rank}")
        arr = sp.MutableDenseNDimArray(list(comps)).reshape(*((size,) * rank))
        T = Tensor(arr.as_immutable(), name=f"T{rank}")
        Q = T.stf()
        assert Q.rank == rank and Q.symmetry == "stf"
        for (i, j) in ((k, k + 1) for k in range(rank - 1)):
            assert Q.is_symmetric(i, j), f"rank={rank}: не симметричен по ({i},{j})"
        assert Q.is_traceless(0, 1), f"rank={rank}: след (0,1) не нулевой"
        assert Q.stf().equals(Q)


def test_stf_is_idempotent():
    """Повторная STF-проекция ничего не меняет (проектор)."""
    comps = sp.symbols("g0:27")
    T = Tensor(sp.MutableDenseNDimArray(list(comps)).reshape(3, 3, 3).as_immutable(), name="T")
    Q = T.stf()
    assert Q.stf().equals(Q)


# ---------------------------------------------------------------------------
# Метрика, подъём/опускание, Levi-Civita
# ---------------------------------------------------------------------------


def test_metric_is_kronecker_and_index_actions_are_identity():
    """Евклидова метрика = дельта; подъём/опускание индекса — тождественны."""
    g = metric(3)
    assert g.equals(Tensor.delta(3))
    T = Tensor([[[a, b], [c, d]], [[d, c], [b, a]]], name="T")
    assert T.raise_index(1).equals(T)
    assert T.lower_index(0).equals(T)

    # с нетривиальной «метрикой»: g = 2*delta -> поднятие умножает на 2
    two_g = Tensor.delta(2) * 2
    v = Tensor([a, b], name="v")
    raised = v.outer(two_g).contract(0, 2)
    assert raised.equals(v * 2)


def test_levi_civita_identities():
    """Тождества Леви-Чивиты: ``eps_ijk eps_ijm = 2 delta_km``, ``eps_ijk eps_ijk = 6``."""
    eps = levi_civita(3)
    assert eps.is_antisymmetric(0, 1) and eps.is_antisymmetric(1, 2)
    prod = eps.outer(eps)
    double = prod.contract(0, 3).contract(0, 2)  # eps_ijk eps_ijm -> ось k, m
    assert double.equals(Tensor.delta(3) * 2)
    full = eps.outer(eps).contract(0, 3).contract(0, 2).contract(0, 1)
    assert sp.simplify(full.as_scalar() - 6) == 0


# ---------------------------------------------------------------------------
# Комплексные: сопряжение, Re/Im, соглашения
# ---------------------------------------------------------------------------


def test_conjugate_and_real_imag_parts():
    """Покомпонентное сопряжение, выделение Re/Im."""
    re1, im1 = sp.symbols("re1 im1", real=True)
    v = Tensor([re1 + sp.I * im1, re1 - sp.I * im1, 3 * re1], name="v")
    conj = v.conjugate()
    assert conj[0] == sp.conjugate(re1 + sp.I * im1)
    assert v.equals(conj.conjugate())
    rp = v.real_part()
    ip = v.imag_part()
    assert all(sp.simplify(x) == 0 for x in rp.imag_part())
    assert all(sp.simplify(x) == 0 for x in ip.imag_part())
    assert (rp + sp.I * ip).equals(v.expand())


def test_complex_pair_and_physical_field_rule():
    """Соглашение ``c = a - I b`` и правило ``B(r,t) = Re[c e^{I omega t}]``."""
    ax, bx = sp.symbols("a_x b_x", real=True)
    cexpr = complex_pair(ax, bx)
    assert sp.simplify(sp.re(cexpr) - ax) == 0
    assert sp.simplify(sp.im(cexpr) + bx) == 0
    # при t=0 физический сигнал равен первой квадратуре
    assert sp.simplify(real_signal(cexpr).subs(TIME, 0) - ax) == 0
    # полная развёртка: Re[(a - I b)(cos + I sin)] = a cos + b sin
    sig = real_signal(cexpr)
    ref = ax * sp.cos(OMEGA * TIME) + bx * sp.sin(OMEGA * TIME)
    assert sp.simplify(sig - ref) == 0


def test_demod_phase_is_conjugate_of_time_phase():
    """Множитель демодуляции --- сопряжённый к временнóму: ``exp(-I m omega t)``."""
    for m in (-1, 0, 1, 2):
        # демодулятор --- точный обратный множитель к временнóму в степени m
        assert sp.simplify(demod_phase(m) * sp.exp(sp.I * m * OMEGA * TIME) - 1) == 0
        assert sp.simplify(sp.conjugate(demod_phase(m)) - sp.exp(sp.I * m * OMEGA * TIME)) == 0


def test_complex_moment_matches_plan_definition():
    """Комплексный момент ``m = m0 (e1 - I e2)`` в декартовых компонентах."""
    m0 = sp.Symbol("m0", positive=True)
    m = complex_moment(m0)
    assert m.rank == 1
    assert m[0] == m0 and m[1] == -sp.I * m0 and m[2] == 0
    # и в произвольных осях
    e1 = sp.Matrix([1, 0, 0])
    e2 = sp.Matrix([0, 0, 1])
    m2 = complex_moment(m0, e1, e2)
    assert m2[0] == m0 and m2[2] == -sp.I * m0


# ---------------------------------------------------------------------------
# Координаты и базисы
# ---------------------------------------------------------------------------


def test_cartesian_spherical_roundtrip_r_squared():
    """Переход декартовы <-> сферические: ``x^2 + y^2 + z^2 = r^2``."""
    expr = cartesian_to_spherical(X ** 2 + Y ** 2 + Z ** 2)
    assert sp.simplify(expr - R ** 2) == 0
    back = spherical_to_cartesian(R ** 2, keep_r=False)
    assert sp.simplify(back - (X ** 2 + Y ** 2 + Z ** 2)) == 0


def test_position_tensor_and_eliminate_r():
    """Вектор положения и устранение ``r``."""
    pos = position_tensor()
    assert pos.rank == 1 and pos[2] == Z
    r2 = eliminate_r(R ** 2)
    assert sp.simplify(r2 - (X ** 2 + Y ** 2 + Z ** 2)) == 0
    assert sp.simplify(eliminate_r(1 / R) - 1 / sp.sqrt(X ** 2 + Y ** 2 + Z ** 2)) == 0


def test_spherical_frame_is_orthonormal():
    """Сферический базис ортонормирован и прав: ``M^T M = I``, ``det M = 1``."""
    M = spherical_frame_matrix()
    assert sp.simplify((M.T * M - sp.eye(3)).applyfunc(sp.simplify)) == sp.zeros(3, 3)
    assert sp.simplify(M.det() - 1) == 0
    # e_r совпадает с первым столбцом матрицы базиса
    assert sp.simplify((unit_r() - M[:, 0]).applyfunc(sp.simplify)) == sp.zeros(3, 1)
    # нормировка остальных ортов
    for u in (unit_theta(), unit_phi()):
        assert sp.simplify((u.T * u)[0] - 1) == 0
    assert sp.simplify((unit_r().T * unit_theta())[0]) == 0
    assert sp.simplify((unit_r().T * unit_phi())[0]) == 0


def test_cyclic_basis_is_unitary_and_consistent():
    """Циклический базис: унитарный переход, ортонормированность по эрмитовой метрике."""
    S = cyclic_to_cartesian_matrix()
    assert sp.simplify((S.adjoint() * S - sp.eye(3)).applyfunc(sp.simplify)) == sp.zeros(3, 3)
    for s in (1, 0, -1):
        e = cyclic_basis_vector(s)
        assert sp.simplify((e.adjoint() * e)[0] - 1) == 0
    # конкретные значения (соглашение пакета)
    sq2 = sp.sqrt(2)
    assert sp.simplify(cyclic_basis_vector(1) - sp.Matrix([-1, -sp.I, 0]) / sq2) == sp.zeros(3, 1)
    assert sp.simplify(cyclic_basis_vector(-1) - sp.Matrix([1, -sp.I, 0]) / sq2) == sp.zeros(3, 1)
    assert cyclic_basis_vector(0) == sp.Matrix([0, 0, 1])
    # переход туда-обратно
    v = sp.Matrix([a, b, c])
    from mag_nav.symbolic.core.coords import to_cartesian_components, to_cyclic_components
    assert sp.simplify(to_cartesian_components(to_cyclic_components(v)) - v) == sp.zeros(3, 1)


def test_basis_tensor_coercion():
    """``basis_tensor`` приводит Matrix / список / Tensor / скаляр к тензору."""
    assert basis_tensor(sp.Matrix([a, b, c])).rank == 1
    assert basis_tensor([a, b]).shape == (2,)
    assert basis_tensor(Tensor([a])).rank == 1
    assert basis_tensor(a).rank == 0
    with pytest.raises(TypeError):
        basis_tensor({"не": "вектор"})


# ---------------------------------------------------------------------------
# Подстановка, упрощение, дифференцирование
# ---------------------------------------------------------------------------


def test_subs_and_simplify_and_diff():
    """``subs``/``simplify``/``diff``/``grad`` покомпонентно."""
    T = Tensor([a * p, b + p, c ** 2], name="T")
    S = T.subs({p: 2})
    assert S[0] == 2 * a and S[1] == b + 2
    U = Tensor([R ** 2 / R], name="U")
    assert sp.simplify(U[0] - R) == 0
    f = Tensor([sp.sin(R * p), R ** 3], name="f")
    d = f.diff(R)
    assert sp.simplify(d[1] - 3 * R ** 2) == 0
    g = Tensor.scalar(X ** 2 * Y + Z * sp.sin(Y), name="phi").grad((X, Y, Z))
    assert g.rank == 1
    assert sp.simplify(g[0] - 2 * X * Y) == 0
    assert sp.simplify(g[2] - sp.sin(Y)) == 0


def test_scalar_arithmetic():
    """Арифметика со скалярами и операторы."""
    v = Tensor([a, b], name="v")
    assert (v * 2)[0] == 2 * a
    assert (2 * v)[1] == 2 * b
    assert (v / 2)[0] == a / 2
    assert (-v)[0] == -a
    assert (v + v).equals(v * 2)
    assert (v - v).equals(Tensor.zeros((2,)))
    with pytest.raises(ValueError):
        v + Tensor([a, b, c])


# ---------------------------------------------------------------------------
# Валидация индексов (нужна этапу 1)
# ---------------------------------------------------------------------------


def test_validate_lm_rules():
    """``validate_lm`` ловит запрещённые пары (l, m) и пропускает символьный l."""
    validate_lm(2, -2)
    validate_lm(sp.Symbol("l"), 1)  # символьный l — можно
    with pytest.raises(ValueError):
        validate_lm(2, 3)
    with pytest.raises(ValueError):
        validate_lm(-1, 0)
    with pytest.raises(ValueError):
        validate_lm(2, sp.Rational(1, 2))
    assert is_concrete_integer(sp.Integer(3)) and not is_concrete_integer(sp.Symbol("l"))


def test_conventions_registry_is_self_describing():
    """Реестр соглашений доступен для печати (единственный источник истины)."""
    text = describe_conventions()
    assert "time_dependence" in text and "condon" in text.lower()
    assert "mu_0" in str(MU0_OVER_4PI)
