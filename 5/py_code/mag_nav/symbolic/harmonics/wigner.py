"""Матрицы Вигнера, повороты моментов и коэффициенты Клебша — Гордана.

Соглашения (фиксируются здесь, ср. :mod:`mag_nav.symbolic.core.conventions`):

* Углы Эйлера — классическая ``z-y-z`` схема:
  :math:`\\mathcal R = R_z(\\alpha)R_y(\\beta)R_z(\\gamma)`.
* Матрица Вигнера:
  :math:`D^{(\\ell)}_{m'm}(\\alpha,\\beta,\\gamma)
  = e^{-i m'\\alpha}\\,d^{(\\ell)}_{m'm}(\\beta)\\,e^{-i m\\gamma}`,
  где ``d`` — малая матрица Вигнера (``sympy.physics.wigner.wigner_d_small``).
* Порядок строк/столбцов матрицы :math:`D^{(\\ell)}` — **по убыванию** ``m``:
  ``m = l, l-1, ..., -l`` (ровно как в ``wigner_d_small``); список индексов
  даёт :func:`m_order`.
* Активный поворот источника на :math:`\\mathcal R` пересчитывает моменты как
  :math:`q^{\\text{lab}}_{\\ell m} = \\sum_{m'} D^{(\\ell)}_{m m'}(\\mathcal R)
  \\,q^{\\text{body}}_{\\ell m'}`. Частный случай поворота вокруг ``z`` на угол
  ``omega*t`` даёт :math:`q_{\\ell m}(t)=e^{-i m\\omega t}\\tilde q_{\\ell m}` —
  тождество проверяется тестом, и именно оно задаёт правило отбора по ``m``.

Для символьного ``l`` явные матрицы недоступны (``wigner_d_small`` требует
конкретного момента), поэтому предусмотрен невычисленный элемент
:func:`wigner_D_element_symbolic` — «следующий этап» для вращений (этап 5).
"""

from __future__ import annotations

import sympy as sp
from sympy.physics.wigner import clebsch_gordan as _clebsch_gordan
from sympy.physics.wigner import wigner_3j, wigner_d_small

from mag_nav.symbolic.core.conventions import I, is_concrete_integer, validate_lm
from mag_nav.symbolic.core.tensor import Tensor

__all__ = [
    "m_order", "m_values",
    "wigner_d_small_matrix", "wigner_D_matrix", "wigner_D", "wigner_D_element",
    "wigner_D_element_symbolic", "wigner_D_conjugate",
    "axial_phase", "rotation_matrix_z", "rotation_matrix_y", "rotation_matrix_x",
    "rotation_matrix_zyz", "euler_zyz_from_matrix", "rotate_moment",
    "clebsch_gordan", "wigner_3j",
]

#: Невычисленный символ элемента матрицы Вигнера (для символьного l).
_D_function = sp.Function("D")


def m_order(l) -> list:
    """Порядок индексов ``m`` в матрицах этого модуля: ``l, l-1, ..., -l``."""
    if not is_concrete_integer(l):
        raise ValueError(f"m_order: l должен быть конкретным целым, получено {l}")
    ell = int(l)
    return list(range(ell, -ell - 1, -1))


def m_values(l) -> list:
    """Допустимые ``m`` по возрастанию: ``-l, ..., l``."""
    if not is_concrete_integer(l):
        raise ValueError(f"m_values: l должен быть конкретным целым, получено {l}")
    ell = int(l)
    return list(range(-ell, ell + 1))


# ---------------------------------------------------------------------------
# Малая и полная матрицы Вигнера
# ---------------------------------------------------------------------------


def wigner_d_small_matrix(l, beta) -> sp.Matrix:
    """Малая матрица Вигнера :math:`d^{(\\ell)}(\\beta)` (вещественная, ортогональная).

    Параметры
    ----------
    l : int
        Целый момент (полученные здесь применения — целочисленные).
    beta : sympy выражение
        Угол поворота вокруг «линии нод» (второй угол Эйлера).

    Возвращает
    ----------
    sympy.Matrix
        Матрица ``(2l+1) x (2l+1)``, элемент ``[i, j]`` соответствует
        ``m' = l - i``, ``m = l - j``.
    """
    if not is_concrete_integer(l):
        raise ValueError("wigner_d_small_matrix требует конкретного целого l")
    return sp.Matrix(wigner_d_small(sp.Integer(int(l)), sp.sympify(beta)))


def wigner_D_matrix(l, alpha, beta, gamma) -> sp.Matrix:
    """Полная матрица Вигнера :math:`D^{(\\ell)}(\\alpha,\\beta,\\gamma)` как ``Matrix``.

    Строится из малой матрицы фазовыми множителями:
    :math:`D_{m'm} = e^{-i m'\\alpha} d_{m'm}(\\beta) e^{-i m\\gamma}`.
    """
    ms = m_order(l)
    d = wigner_d_small_matrix(l, beta)
    n = len(ms)
    out = sp.MutableDenseMatrix(n, n, lambda i, j: sp.simplify(
        sp.exp(-I * ms[i] * alpha) * d[i, j] * sp.exp(-I * ms[j] * gamma)))
    return out.applyfunc(sp.simplify)


def wigner_D(l, alpha, beta, gamma, name: str = "D") -> Tensor:
    """Матрица Вигнера как :class:`Tensor` ранга 2 (индексы ``m'``, ``m``).

    Возвращает
    ----------
    Tensor
        Форма ``(2l+1, 2l+1)``, ``indices=('m_p', 'm')``. Осевый порядок —
        :func:`m_order` (``m`` по убыванию).
    """
    mat = wigner_D_matrix(l, alpha, beta, gamma)
    return Tensor.from_matrix(mat, name=f"{name}^{l}", indices=("m_p", "m"))


def wigner_D_element(l, mp, m, alpha, beta, gamma):
    """Отдельный элемент :math:`D^{(\\ell)}_{m'm}` (конкретный ``l``).

    Параметры
    ----------
    l, mp, m : int
        Момент и его проекции (``|mp|, |m| <= l``).
    alpha, beta, gamma
        Углы Эйлера ``z-y-z``.

    Возвращает
    ----------
    sympy.Expr
        Значение элемента.
    """
    validate_lm(l, mp)
    validate_lm(l, m)
    ms = m_order(l)
    if mp not in ms or m not in ms:
        raise ValueError(f"wigner_D_element: индексы ({mp}, {m}) вне {ms}")
    return wigner_D_matrix(l, alpha, beta, gamma)[ms.index(mp), ms.index(m)]


def wigner_D_element_symbolic(l, mp, m, alpha, beta, gamma):
    """Невычисленный элемент :math:`D^{(\\ell)}_{m'm}` при символьном ``\\ell``.

    Возвращает ``D(l, m', m, alpha, beta, gamma)`` как символ ``Function`` —
    «держатель» для формул произвольного порядка; вычисляется подстановкой
    конкретного ``l`` (см. :func:`wigner_D_element`).
    """
    return _D_function(l, mp, m, alpha, beta, gamma)


def wigner_D_conjugate(l, alpha, beta, gamma) -> sp.Matrix:
    """Эрмитово сопряжение :math:`D^{(\\ell)\\dagger}` (обратному повороту)."""
    return wigner_D_matrix(l, alpha, beta, gamma).adjoint()


def axial_phase(l, m, omega=sp.Symbol("omega", real=True), t=sp.Symbol("t", real=True)):
    """Фазовый множитель аксиального вращения: :math:`e^{-i m\\omega t}`.

    Физический смысл
    ----------------
    Источник, вращающийся вокруг оси ``z`` с частотой ``omega``, в лабораторной
    системе имеет моменты :math:`q_{\\ell m}(t) = e^{-i m\\omega t}\\tilde q_{\\ell m}`.
    Это же — угловая часть матрицы Вигнера для ``R = Rz(omega*t)`` (тест
    :func:`rotate_moment` сверяет оба выражения).
    """
    return sp.exp(-I * sp.sympify(m) * omega * t)


# ---------------------------------------------------------------------------
# Матрицы поворота в декартовом базисе
# ---------------------------------------------------------------------------

_ALPHA = sp.Symbol("alpha", real=True)
_BETA = sp.Symbol("beta", real=True)
_GAMMA = sp.Symbol("gamma", real=True)


def rotation_matrix_z(angle) -> sp.Matrix:
    """Поворот вокруг ``z`` на угол ``angle`` (активный, правый)."""
    c, s = sp.cos(angle), sp.sin(angle)
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix_y(angle) -> sp.Matrix:
    """Поворот вокруг ``y`` на угол ``angle``."""
    c, s = sp.cos(angle), sp.sin(angle)
    return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_x(angle) -> sp.Matrix:
    """Поворот вокруг ``x`` на угол ``angle``."""
    c, s = sp.cos(angle), sp.sin(angle)
    return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_zyz(alpha=_ALPHA, beta=_BETA, gamma=_GAMMA) -> sp.Matrix:
    """Декартова матрица поворота :math:`\\mathcal R = R_z(\\alpha)R_y(\\beta)R_z(\\gamma)`.

    Именно эта схема углов Эйлера согласована с ``wigner_d_small``.
    """
    R = (rotation_matrix_z(alpha) * rotation_matrix_y(beta) * rotation_matrix_z(gamma))
    return R.applyfunc(sp.expand_trig)


def euler_zyz_from_matrix(R) -> tuple:
    """Углы Эйлера ``z-y-z`` из декартовой матрицы поворота (3x3).

    Возвращает
    ----------
    tuple[sympy.Expr, sympy.Expr, sympy.Expr]
        ``(alpha, beta, gamma)`` — ветка для ``sin(beta) != 0``; при ``beta = 0``
        или ``pi`` разложение вырождено (углы определяются с точностью до
        ``alpha ± gamma``), что оставлено на усмотрение вызывающему коду.

    Формулы
    -------
    :math:`\\beta=\\operatorname{atan2}(\\sqrt{R_{31}^2+R_{32}^2}, R_{33})`,
    :math:`\\alpha=\\operatorname{atan2}(R_{32}, R_{31})`,
    :math:`\\gamma=\\operatorname{atan2}(-R_{13}, R_{23})`.
    """
    M = sp.Matrix(R)
    if M.shape != (3, 3):
        raise ValueError(f"euler_zyz_from_matrix: ожидается 3x3, получено {M.shape}")
    alpha = sp.atan2(M[2, 1], M[2, 0])
    beta = sp.atan2(sp.sqrt(M[2, 0] ** 2 + M[2, 1] ** 2), M[2, 2])
    gamma = sp.atan2(-M[0, 2], M[1, 2])
    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# Пересчёт набора моментов при повороте
# ---------------------------------------------------------------------------


def _as_euler(R, euler) -> tuple:
    """Приводит задание поворота к triple углов Эйлера ``z-y-z``."""
    if euler is not None:
        a, b, g = (sp.sympify(v) for v in euler)
        return a, b, g
    if R is None:
        raise ValueError("rotate_moment: нужен либо R (матрица 3x3), либо euler=(alpha,beta,gamma)")
    M = sp.Matrix(R)
    if M.shape != (3, 3):
        raise ValueError(f"rotate_moment: R должна быть 3x3, получено {M.shape}")
    return euler_zyz_from_matrix(M)


def rotate_moment(q_dict: dict, R=None, euler=None, inverse: bool = False,
                  group_key=None) -> dict:
    """Пересчёт набора моментов :math:`\\{q_{\\ell m}\\}` при повороте источника.

    Параметры
    ----------
    q_dict : dict
        Словарь ``{(l, m): значение}``.
    R : sympy.Matrix | None
        Декартова матрица поворота 3x3 (углы извлекаются :func:`euler_zyz_from_matrix`).
    euler : tuple | None
        Углы Эйлера ``(alpha, beta, gamma)`` схемы ``z-y-z`` — альтернатива ``R``.
    inverse : bool
        Если ``True`` — поворот против :math:`\\mathcal R` (то есть
        :math:`D^{(\\ell)\\dagger}`); удобно для перехода «лаборатория -> корпус».
    group_key : callable | None
        Функция ``key -> group``, задающая, какие ключи смешиваются поворотом.
        По умолчанию — момент ``l`` (стандартное поведение irreducible tensor).

    Возвращает
    ----------
    dict
        Новый словарь ``{(l, m): символьное выражение}`` в повёрнутой системе.

    Закон преобразования
    --------------------
    :math:`q^{\\text{lab}}_{\\ell m} = \\sum_{m'} D^{(\\ell)}_{m m'}(\\mathcal R)
    q^{\\text{body}}_{\\ell m'}`; при ``inverse=True`` используется
    :math:`D^{(\\ell)\\dagger}`.
    """
    alpha, beta, gamma = _as_euler(R, euler)
    key_of = (lambda k: k[0]) if group_key is None else group_key

    groups: dict = {}
    for key, value in q_dict.items():
        groups.setdefault(key_of(key), []).append(key)

    out: dict = {}
    for group, keys in groups.items():
        ls = keys[0][0]
        ms = sorted({k[1] for k in keys}, reverse=True)
        if not is_concrete_integer(ls):
            raise ValueError("rotate_moment: символьный l не поддерживается — "
                             "используйте wigner_D_element_symbolic (этап 5)")
        D = wigner_D_matrix(ls, alpha, beta, gamma)
        order = m_order(ls)
        if inverse:
            D = D.adjoint()
        value_by_m = {k[1]: sp.sympify(v) for k, v in q_dict.items() if k[0] == group}
        for m in ms:
            acc = sp.Integer(0)
            for mp in order:
                if mp in value_by_m:
                    elem = D[order.index(m), order.index(mp)]
                    acc = acc + elem * value_by_m[mp]
            out[(ls, m)] = sp.simplify(acc)
    return out


# ---------------------------------------------------------------------------
# Коэффициенты Клебша — Гордана
# ---------------------------------------------------------------------------


def clebsch_gordan(l1, m1, l2, m2, L=None, M=None):
    """Коэффициенты Клебша — Гордана :math:`\\langle l_1 m_1; l_2 m_2 | L M\\rangle`.

    Параметры
    ----------
    l1, m1, l2, m2 : int
        Сцепляемые моменты и их проекции.
    L, M : int | None
        Итоговый момент и проекция. Если оба ``None`` — возвращается **словарь**
        ``{(L, M): coefficient}`` по всем допустимым ``|L|`` (треугольное условие
        ``|l1-l2| <= L <= l1+l2`` и ``M = m1 + m2``), нулевые значения опущены.
        Если задан только ``L`` — проецируется на максимальную допустимую ``M``.

    Возвращает
    ----------
    sympy.Expr | dict
        Один коэффициент либо полный набор (символьно-точный, выражается через
        факториалы и sqrt).

    Зачем нужно
    -----------
    Связь декартовых и сферических компонент: тензорное произведение векторов
    раскладывается по :math:`\\ell` именно этими коэффициентами (например,
    :math:`1\\otimes1 = 0\\oplus1\\oplus2` — след, векторное, STF-части).
    """
    if L is None and M is None:
        result: dict = {}
        for Lc in range(abs(l1 - l2), l1 + l2 + 1):
            Mc = m1 + m2
            if abs(Mc) > Lc:
                continue
            coef = sp.nsimplify(_clebsch_gordan(l1, l2, Lc, m1, m2, Mc))
            if coef != 0:
                result[(Lc, Mc)] = coef
        return result
    if M is None:
        M = m1 + m2
    return sp.nsimplify(_clebsch_gordan(l1, l2, L, m1, m2, M))
