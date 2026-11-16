"""Сферические гармоники :math:`Y_{\\ell m}` и их производные.

Конвенция (см. :mod:`mag_nav.symbolic.core.conventions`): ортонормированные
комплексные гармоники с фазой Кондона — Шортли,
:math:`Y_{\\ell m}(\\theta,\\varphi)=
\\sqrt{\\frac{2\\ell+1}{4\\pi}\\frac{(\\ell-m)!}{(\\ell+m)!}}
P_\\ell^{m}(\\cos\\theta)\\,e^{i m\\varphi}`, аргументы — ``(theta, phi)``
(``theta`` — полярный угол). Реализация — ``sympy.Ynm``.

Порядок ``l`` может оставаться **символьным**: дифференцирование по ``phi``
даёт :math:`i m Y_{\\ell m}`, а по ``theta`` — лестничное соотношение
:math:`\\partial_\\theta Y_{\\ell m} = m\\cot\\theta Y_{\\ell m}
+ \\sqrt{(\\ell-m)(\\ell+m+1)}e^{-i\\varphi}Y_{\\ell,m+1}`,
которое SymPy выводит для произвольного ``l``. Конкретные порядки получаются
подстановкой ``l -> Integer``.

Примечание об «аренде»: функции :func:`physical_gradient_spherical` и
:func:`rotate_to_cartesian_frame` — это геометрия кадра, необходимая уже здесь.
На этапе 3 их общие версии (``grad``, ``div``, ``curl``, ``laplacian``) переедут
в :mod:`mag_nav.symbolic.calculus.operators`, а отсюда будут реэкспортироваться.
"""

from __future__ import annotations

import sympy as sp
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.functions.special.polynomials import assoc_legendre

from mag_nav.symbolic.core.conventions import I, PI, is_concrete_integer, validate_lm
from mag_nav.symbolic.core.coords import (
    PHI,
    R,
    THETA,
    spherical_frame_matrix,
    spherical_to_cartesian,
)
from mag_nav.symbolic.core.tensor import Tensor

__all__ = [
    "Y", "Y_explicit", "Y_general", "Y_conjugate_relation",
    "dY_dtheta", "dY_dphi", "dY_dtheta_down_ladder", "ladder_coefficient",
    "grad_Y", "physical_gradient_spherical", "rotate_to_cartesian_frame",
    "surface_gradient", "laplace_beltrami",
    "orthogonality_integral", "check_orthogonality", "norm_integral",
    "m_values",
]


def m_values(l: int) -> list:
    """Допустимые азимутальные индексы ``m = -l, ..., l`` (по возрастанию).

    Параметры
    ----------
    l : int
        Порядок мультиполя (конкретное целое).

    Возвращает
    ----------
    list[int]
        Список ``m``.
    """
    if not is_concrete_integer(l):
        raise ValueError(f"m_values: l должен быть конкретным целым, получено {l}")
    ell = int(l)
    return list(range(-ell, ell + 1))


# ---------------------------------------------------------------------------
# Сами гармоники
# ---------------------------------------------------------------------------


def Y(l, m):
    """Символьное выражение :math:`Y_{\\ell m}(\\theta,\\varphi)`.

    Параметры
    ----------
    l, m
        Порядок и азимутальный индекс. ``l`` может быть символом — тогда
        возвращается *невычисленное* ``Ynm(l, m, theta, phi)``, с которым
        корректно работают производные (лестничные рекурсии).

    Возвращает
    ----------
    sympy.Expr
        Гармоника; для конкретных ``l, m`` — упрощённое явное выражение.

    Физический смысл
    ----------------
    Базисная функция на единичной сфере: :math:`\\int Y_{\\ell m}^* Y_{\\ell' m'}
    d\\Omega = \\delta_{\\ell\\ell'}\\delta_{mm'}`.
    """
    validate_lm(l, m)
    if is_concrete_integer(l):
        return sp.simplify(Ynm(int(l), int(m), THETA, PHI).expand(func=True))
    return Ynm(l, m, THETA, PHI)


def Y_explicit(l, m):
    """Явная форма :math:`Y_{\\ell m}` через :math:`\\cos\\theta` и :math:`e^{i m\\varphi}`.

    Отличается от :func:`Y` тем, что дополнительно раскладывает кратные углы
    (``sin(2 theta)`` -> ``2 sin(theta) cos(theta)``) и приводит выражение к
    виду «полином от ``sin(theta)``, ``cos(theta)``, ``exp(I*m*phi)``» — именно
    в таком виде оно переводится в декартовы координаты.

    Параметры
    ----------
    l, m : int
        Конкретные целые порядок и индекс.

    Возвращает
    ----------
    sympy.Expr
        Явное выражение; используется для верификации и отладки.

    Raises
    ------
    ValueError
        Если ``l`` не конкретное целое.
    """
    validate_lm(l, m)
    if not is_concrete_integer(l):
        raise ValueError("Y_explicit требует конкретного целого l (для символьного l используйте Y)")
    e = sp.simplify(Ynm(int(l), int(m), THETA, PHI).expand(func=True))
    return sp.expand_trig(e)


def Y_general(l, m):
    """Замкнутая формула для произвольного ``l`` через присоединённый Лежандр.

    :math:`Y_{\\ell m} = \\sqrt{\\frac{2\\ell+1}{4\\pi}\\frac{(\\ell-m)!}{(\\ell+m)!}}
    P_\\ell^{m}(\\cos\\theta)e^{i m\\varphi}` — записана явно через
    ``assoc_legendre`` и ``gamma``/``factorial``, поэтому пригодна для
    аналитических выкладок при символьном ``l`` (в отличие от невычисленного
    ``Ynm``). Дифференцировать её по ``theta`` невыгодно: производная
    ``assoc_legendre`` не сворачивается в лестничную рекурсию автоматически.

    Параметры
    ----------
    l, m
        Порядок (можно символьный, ``>= 0``) и целый индекс.

    Возвращает
    ----------
    sympy.Expr
        Явное выражение с ``assoc_legendre(l, m, cos(theta))``.
    """
    validate_lm(l, m)
    norm = sp.sqrt((2 * l + 1) / (4 * PI) * sp.gamma(l - m + 1) / sp.gamma(l + m + 1))
    expr = sp.simplify(norm * assoc_legendre(l, m, sp.cos(THETA)) * sp.exp(I * m * PHI))
    # theta в [0, pi] => sin(theta) >= 0: снимаем Abs, который SymPy порождает из
    # sqrt(1 - cos(theta)**2) внутри assoc_legendre.
    expr = expr.replace(lambda e: isinstance(e, sp.Abs) and e.args[0] == sp.sin(THETA),
                        lambda e: sp.sin(THETA))
    return expr


def Y_conjugate_relation(l, m):
    """Сопряжение: :math:`\\overline{Y_{\\ell m}} = (-1)^m Y_{\\ell,-m}` (правая часть)."""
    return (-1) ** sp.sympify(m) * Y(l, -m)


# ---------------------------------------------------------------------------
# Производные и лестничные рекурсии
# ---------------------------------------------------------------------------


def dY_dtheta(l, m):
    """:math:`\\partial_\\theta Y_{\\ell m}` — лестничная рекурсия «вверх» (произвольный ``l``).

    Формула: :math:`\\partial_\\theta Y_{\\ell m} = m\\cot\\theta\\,Y_{\\ell m}
    + \\sqrt{(\\ell-m)(\\ell+m+1)}\\,e^{-i\\varphi}Y_{\\ell,m+1}`.
    """
    return sp.diff(Y(l, m), THETA)


def dY_dtheta_down_ladder(l, m):
    """:math:`\\partial_\\theta Y_{\\ell m}` — альтернативная рекурсия «вниз» (произвольный ``l``).

    Формула: :math:`\\partial_\\theta Y_{\\ell m} =
    -m\\cot\\theta\\,Y_{\\ell m} - \\sqrt{(\\ell+m)(\\ell-m+1)}\\,e^{+i\\varphi}Y_{\\ell,m-1}`.
    Эквивалентность :func:`dY_dtheta` проверяется тестом (для конкретных ``l, m``).
    """
    return (-m * sp.cot(THETA) * Y(l, m)
            - sp.sqrt((l + m) * (l - m + 1)) * sp.exp(I * PHI) * Y(l, m - 1))


def dY_dphi(l, m):
    """:math:`\\partial_\\varphi Y_{\\ell m} = i m Y_{\\ell m}` (произвольный ``l``)."""
    return sp.diff(Y(l, m), PHI)


def ladder_coefficient(l, m):
    """Коэффициент лестничного перехода :math:`\\sqrt{(\\ell-m)(\\ell+m+1)}`."""
    return sp.sqrt((l - m) * (l + m + 1))


# ---------------------------------------------------------------------------
# Геометрия кадра (переедет в calculus/operators.py на этапе 3)
# ---------------------------------------------------------------------------


def physical_gradient_spherical(expr, name: str = "grad") -> Tensor:
    """Градиент скаляра в **физических** компонентах сферического ортонормированного кадра.

    :math:`\\nabla f = \\left(\\partial_r f,\\; \\frac{1}{r}\\partial_\\theta f,\\;
    \\frac{1}{r\\sin\\theta}\\partial_\\varphi f\\right)`.

    Параметры
    ----------
    expr : sympy выражение
        Скалярное поле ``f(r, theta, phi)``.
    name : str
        Имя результирующего тензора.

    Возвращает
    ----------
    Tensor
        Ранг 1, форма ``(3,)``, оси соответствуют ``(r, theta, phi)``-ортам.
    """
    f = sp.sympify(expr)
    comps = [
        sp.diff(f, R),
        sp.diff(f, THETA) / R,
        sp.diff(f, PHI) / (R * sp.sin(THETA)),
    ]
    return Tensor([sp.simplify(c) for c in comps], name=name, indices=("i",))


def rotate_to_cartesian_frame(vec: Tensor, name: str = "cart") -> Tensor:
    """Поворот вектора из сферического кадра в декартов: ``v_cart = M @ v_sph``.

    Параметры
    ----------
    vec : Tensor
        Ранг 1 с физическими компонентами в базисе ``(e_r, e_theta, e_phi)``.
    name : str
        Имя результата.

    Возвращает
    ----------
    Tensor
        Ранг 1, декартовы компоненты (вещественная ортогональная матрица кадра,
        значит закон преобразования фаз сохраняется).
    """
    if vec.rank != 1 or vec.shape != (3,):
        raise ValueError(f"rotate_to_cartesian_frame: ожидается вектор формы (3,), получено {vec.shape}")
    M = spherical_frame_matrix()
    comps = [sp.simplify(sum(M[i, j] * vec[j] for j in range(3))) for i in range(3)]
    return Tensor(comps, name=name, indices=("i",))


def surface_gradient(l, m, name: str = "grad_Omega") -> Tensor:
    """Поверхностный (угловой) градиент :math:`\\vec\\nabla_\\Omega Y_{\\ell m}`.

    Это ``r``-умноженный угловой блок полного градиента: компоненты
    ``(0, \\partial_\\theta Y, (1/\\sin\\theta)\\partial_\\varphi Y)`` в сферическом
    кадре. Удобно для оценки угловой структуры поля.
    """
    comps = [sp.Integer(0), sp.simplify(dY_dtheta(l, m)), sp.simplify(dY_dphi(l, m) / sp.sin(THETA))]
    return Tensor(comps, name=name, indices=("i",))


def grad_Y(l, m, frame: str = "spherical", name: str = "grad_Y") -> Tensor:
    """Градиент :math:`\\nabla Y_{\\ell m}` как функция точки наблюдения.

    Параметры
    ----------
    l, m
        Порядок (можно символьный) и азимутальный индекс.
    frame : {'spherical', 'cartesian', 'angular'}
        Базис компонент. ``spherical`` — физические компоненты в кадре
        ``(e_r, e_theta, e_phi)``; ``cartesian`` — те же компоненты, повёрнутые
        в базис ``(e_x, e_y, e_z)``; ``angular`` — поверхностный градиент без
        множителя ``1/r``.
    name : str
        Имя результирующего тензора.

    Возвращает
    ----------
    Tensor
        Ранг 1, форма ``(3,)``. Радиальная компонента равна нулю: гармоника
        зависит только от углов.

    Физический смысл
    ----------------
    :math:`\\nabla Y_{\\ell m} = \\frac{1}{r}\\hat e_\\theta
    \\partial_\\theta Y_{\\ell m} + \\frac{i m}{r\\sin\\theta}\\hat e_\\varphi
    Y_{\\ell m}` — угловой «хвост» мультипольного поля (см. ``harmonics.multipole``).
    """
    if frame == "angular":
        return surface_gradient(l, m, name=name)
    g = physical_gradient_spherical(Y(l, m), name=f"{name}_sph")
    if frame == "spherical":
        return g
    if frame == "cartesian":
        return rotate_to_cartesian_frame(g, name=name)
    raise ValueError(f"grad_Y: неизвестный кадр {frame!r}")


def laplace_beltrami(expr, name: str = "Delta_Omega") -> Tensor:
    """Угловой лапласиан (оператор Бельтрами) на единичной сфере.

    :math:`\\Delta_\\Omega f = \\frac{1}{\\sin\\theta}\\partial_\\theta
    (\\sin\\theta\\,\\partial_\\theta f) + \\frac{1}{\\sin^2\\theta}
    \\partial_\\varphi^2 f`.

    Возвращает
    ----------
    Tensor
        Ранг 0: скаляр ``Delta_Omega expr``. Собственное значение на
        :math:`Y_{\\ell m}` равно :math:`-\\ell(\\ell+1)Y_{\\ell m}`.
    """
    f = sp.sympify(expr)
    term_theta = sp.diff(sp.sin(THETA) * sp.diff(f, THETA), THETA) / sp.sin(THETA)
    term_phi = sp.diff(f, PHI, 2) / sp.sin(THETA) ** 2
    return Tensor.scalar(sp.simplify(term_theta + term_phi), name=name)


# ---------------------------------------------------------------------------
# Ортогональность и нормировка
# ---------------------------------------------------------------------------


def orthogonality_integral(l1, m1, l2, m2):
    """Скалярное произведение на сфере :math:`\\int Y_{l_1 m_1}^{*} Y_{l_2 m_2}\\,d\\Omega`.

    Интегрирование символьное: сначала по ``phi`` (от ``0`` до ``2 pi``), затем по
    ``theta`` с весовым множителем ``sin(theta)``.

    Параметры
    ----------
    l1, m1, l2, m2 : int
        Конкретные целые порядки и индексы (интегрирование требует явных выражений).

    Возвращает
    ----------
    sympy.Expr
        Значение интеграла (после ``simplify``): ``1`` при совпадении пар и ``0`` иначе.
    """
    for pair in ((l1, m1), (l2, m2)):
        if not is_concrete_integer(pair[0]):
            raise ValueError(f"orthogonality_integral: порядок должен быть конкретным, получено {pair}")
    f1 = sp.conjugate(Y_explicit(l1, m1))
    f2 = Y_explicit(l2, m2)
    integrand = sp.expand_trig(sp.simplify(f1 * f2 * sp.sin(THETA)))
    over_theta = sp.integrate(integrand, (THETA, 0, PI))
    over_phi = sp.integrate(over_theta, (PHI, 0, 2 * PI))
    return sp.simplify(over_phi)


def check_orthogonality(l1, m1, l2, m2) -> bool:
    """Символьная проверка ортонормированности пары гармоник.

    Возвращает
    ----------
    bool
        ``True``, если :math:`\\int Y_{l_1m_1}^* Y_{l_2m_2}d\\Omega`
        равно :math:`\\delta_{l_1l_2}\\delta_{m_1m_2}`.
    """
    value = orthogonality_integral(l1, m1, l2, m2)
    expected = sp.Integer(1) if (l1 == l2 and m1 == m2) else sp.Integer(0)
    return sp.simplify(value - expected) == 0


def norm_integral(l, m):
    """Норма :math:`\\int |Y_{\\ell m}|^2 d\\Omega` (должна равняться 1)."""
    return orthogonality_integral(l, m, l, m)
