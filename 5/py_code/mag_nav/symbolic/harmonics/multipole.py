"""Мультипольный базис: потенциал, поле и декартовые STF-тензоры.

Два эквивалентных описания одного и того же пространства решений уравнения
Лапласа вне источника:

* **Сферическое** — потенциал
  :math:`\\Phi_{\\ell m}(\\mathbf r) = Y_{\\ell m}(\\theta,\\varphi)/r^{\\ell+1}`,
  поле :math:`\\mathbf g_{\\ell m} = -\\nabla\\Phi_{\\ell m}`
  (:func:`multipole_potential`, :func:`multipole_field`). Порядок ``l`` может
  быть символьным; конкретные порядки получаются подстановкой.
* **Декартово STF** — :math:`\\partial_{i_1}\\dots\\partial_{i_\\ell}(1/r)`,
  симметричный бесследовый тензор ранга ``l`` (:func:`cartesian_stf_multipole`).
  Связь: :math:`\\partial_K(1/r) = (-1)^\\ell (2\\ell-1)!!\\,
  \\hat r_{\\langle K\\rangle}/r^{\\ell+1}`.

Коэффициент ``l!`` и знак ``(-1)^l`` в :func:`stf_multipole_potential`
соответствуют разложению :math:`\\Phi(\\mathbf r)=\\sum_\\ell (-1)^\\ell/\\ell!\\,
Q^{K}\\partial_K (1/r)` (мультиполи как симметричные тензоры moments).
"""

from __future__ import annotations

import sympy as sp

from mag_nav.symbolic.core.conventions import I, PI, is_concrete_integer, validate_lm
from mag_nav.symbolic.core.coords import (
    PHI,
    R,
    THETA,
    X,
    Y,
    Z,
    eliminate_r,
    restore_r,
    spherical_to_cartesian,
)
from mag_nav.symbolic.core.tensor import Tensor
from mag_nav.symbolic.harmonics.spherical import (
    Y as Y_lm,  # не путать с координатным символом Y из coords
    physical_gradient_spherical,
    rotate_to_cartesian_frame,
)

__all__ = [
    "CARTESIAN_AXES",
    "multipole_potential",
    "multipole_field",
    "multipole_field_cartesian",
    "cartesian_stf_multipole",
    "stf_radial_tensor",
    "stf_multipole_potential",
    "stf_multipole_field",
    "stf_moment_symbols",
]

#: Оси декартова базиса (порядок компонент во всех тензорах этого модуля).
CARTESIAN_AXES = (X, Y, Z)


# ---------------------------------------------------------------------------
# Сферическое представление: потенциал и поле g_{lm}
# ---------------------------------------------------------------------------


def multipole_potential(l, m, radius: sp.Basic = R):
    """Скалярный мультипольный потенциал :math:`\\Phi_{\\ell m}=Y_{\\ell m}/r^{\\ell+1}`.

    Параметры
    ----------
    l, m
        Порядок (допускается символьный) и азимутальный индекс.
    radius : sympy выражение
        Переменная, играющая роль радиуса (по умолчанию :data:`R`).

    Возвращает
    ----------
    sympy.Expr
        :math:`Y_{\\ell m}(\\theta,\\varphi)\\,r^{-(\\ell+1)}`.

    Физический смысл
    ----------------
    Решение уравнения Лапласа с заданной угловой структурой :math:`Y_{\\ell m}`;
    нормировка на :math:`4\\pi\\varepsilon_0` и моментоподобные множители вносятся
    на уровне прямой модели (этап 2), здесь — чистый базис.
    """
    validate_lm(l, m)
    return sp.simplify(Y_lm(l, m) * sp.Pow(radius, -(l + 1)))


def multipole_field(l, m, frame: str = "cartesian", radius: sp.Basic = R) -> Tensor:
    """Мультипольное поле :math:`\\mathbf g_{\\ell m} = -\\nabla\\Phi_{\\ell m}`.

    Параметры
    ----------
    l, m
        Порядок (можно символьный) и индекс ``|m| <= l``.
    frame : {'cartesian', 'spherical'}
        Базис компонент: ``cartesian`` (по умолчанию, декартовы компоненты) или
        ``spherical`` — физические компоненты в кадре ``(e_r, e_theta, e_phi)``.
    radius : sympy выражение
        Радиальная переменная.

    Возвращает
    ----------
    Tensor
        Ранг 1, форма ``(3,)``. Для символьного ``l`` компоненты записаны через
        :math:`Y_{\\ell m}` и :math:`Y_{\\ell,m\\pm1}` (лестничные рекурсии) —
        замкнутая формула произвольного порядка.

    Физический смысл
    ----------------
    Это тот самый базисный вектор :math:`\\mathbf g_\\ell(\\mathbf r)` из
    прямой модели: поле единицы момента в точке наблюдения. Комплексность
    ``(l, m)`` даёт круговую поляризацию, нужную для правила отбора по ``m``.
    """
    phi = multipole_potential(l, m, radius=radius)
    g_sph = -physical_gradient_spherical(phi, name=f"g_{l}{m}")
    if frame == "spherical":
        return g_sph.simplify(name=f"g_{l}{m}_sph")
    if frame == "cartesian":
        return rotate_to_cartesian_frame(g_sph, name=f"g_{l}{m}").simplify(name=f"g_{l}{m}")
    raise ValueError(f"multipole_field: неизвестный кадр {frame!r}")


def multipole_field_cartesian(l, m, keep_r: bool = True) -> Tensor:
    """То же поле, но компоненты переписаны через ``x, y, z`` (без углов).

    Порядок ``l`` обязан быть конкретным целым: замкнутая декартова форма
    :math:`Y_{\\ell m}` существует только для фиксированного порядка (при
    символьном ``l`` используйте :func:`multipole_field` с ``frame='cartesian'``).

    Параметры
    ----------
    l, m : int
        Конкретные целые.
    keep_r : bool
        Если ``True`` (по умолчанию) радиус остаётся символом ``r``; если
        ``False``, ``r`` устраняется заменой ``sqrt(x^2+y^2+z^2)``.

    Возвращает
    ----------
    Tensor
        Ранг 1 с компонентами от ``x, y, z`` (опционально ``r``).

    Зачем нужно
    -----------
    В такой форме поле непосредственно дифференцируется по ``x, y, z`` — это
    нужно для Якобиана прямой модели (этап 3).
    """
    validate_lm(l, m)
    if not is_concrete_integer(l):
        raise ValueError(
            "multipole_field_cartesian требует конкретного целого l: "
            "замкнутая декартова форма Y_lm существует только при фиксированном порядке"
        )
    g = multipole_field(l, m, frame="cartesian")
    comps = [sp.simplify(spherical_to_cartesian(c, keep_r=True)) for c in g]
    out = Tensor(comps, name=f"g_{l}{m}(x,y,z)")
    if not keep_r:
        out = out.applyfunc(lambda e: eliminate_r(e))
    return out


# ---------------------------------------------------------------------------
# Декартово STF-представление
# ---------------------------------------------------------------------------


def _cartesian_derivative(T: Tensor, name: str) -> Tensor:
    """Производная по декартовым координатам: добавляет ось справа (ранг +1).

    Компоненты берутся в порядке ``CARTESIAN_AXES``; выражения сначала
    освобождаются от символа ``r`` (иначе ``d/dr`` по ``x`` был бы нулём), а
    затем ``r`` возвращается для компактности.
    """
    flat = [eliminate_r(c) for c in T]
    new_flat = [sp.simplify(restore_r(sp.diff(c, axis))) for c in flat for axis in CARTESIAN_AXES]
    shape = tuple(T.shape) + (3,)
    arr = sp.MutableDenseNDimArray(new_flat).reshape(*shape).as_immutable()
    return Tensor(arr, name=name)


def cartesian_stf_multipole(order: int, name: str | None = None) -> Tensor:
    """Декартов мультипольный базис :math:`M^{(\\ell)}_{K}=\\partial_{i_1}\\dots\\partial_{i_\\ell}(1/r)`.

    Параметры
    ----------
    order : int
        Порядок ``l`` (конкретное целое ``>= 0``).
    name : str | None
        Имя результата.

    Возвращает
    ----------
    Tensor
        Ранг ``l`` (для ``l = 0`` — скаляр ``1/r``), форма ``(3,)*l``,
        ``symmetry='stf'``: симметричен по всем осям и бесследов
        (след = лапласиан ``1/r = 0`` вне начала координат).

    Физический смысл
    ----------------
    Тензорное (индексное) альтер-эго сферических гармоник: с ним удобно
    работать, когда моменты заданы как симметричные тензоры (квадруполь ``Q_ij``),
    а не как числа ``q_{lm}``.
    """
    order = int(order)
    if order < 0:
        raise ValueError(f"cartesian_stf_multipole: порядок должен быть >= 0, получено {order}")
    T = Tensor.scalar(1 / R, name="1/r")
    for k in range(order):
        T = _cartesian_derivative(T, name=f"d^{k + 1}(1/r)")
    return Tensor(T, name=name or f"M_{order}", symmetry="stf" if order >= 1 else None)


def stf_radial_tensor(order: int, name: str = "rhat_STF") -> Tensor:
    """Симметричный бесследовый тензор из единичного вектора:
    :math:`\\hat r_{\\langle i_1 \\dots i_\\ell \\rangle}`.

    Строится как ``order``-кратное внешнее произведение ``r_hat`` с полной
    симметризацией и STF-проекцией (компоненты — через ``x, y, z, r``).

    Возвращает
    ----------
    Tensor
        Ранг ``l``, ``symmetry='stf'``.
    """
    order = int(order)
    if order < 0:
        raise ValueError(f"stf_radial_tensor: порядок должен быть >= 0, получено {order}")
    n = Tensor([X / R, Y / R, Z / R], name="r_hat")
    T = Tensor.scalar(sp.Integer(1), name="one")
    for _ in range(order):
        T = T.outer(n)
    if order == 0:
        return T
    return T.stf(name=name)


def stf_moment_symbols(order: int, prefix: str = "Q") -> Tensor:
    """Символьный STF-тензор момента порядка ``order`` (независимые символы).

    Полезно для построения общего выражения поля/Якобиана при фиксированном ``l``
    без привязки к конкретному источнику.

    Параметры
    ----------
    order : int
        Ранг момента.
    prefix : str
        Префикс имён символов.

    Возвращает
    ----------
    Tensor
        Ранг ``order`` с ``3**order`` символами, спроецированными в STF.
    """
    order = int(order)
    if order == 0:
        q0 = sp.Symbol(f"{prefix}", real=True)
        return Tensor.scalar(q0, name=prefix)
    comps = sp.symbols(f"{prefix}_0:{3 ** order}")
    T = Tensor(sp.MutableDenseNDimArray(list(comps)).reshape(*((3,) * order)).as_immutable(),
               name=prefix)
    return T.stf(name=f"{prefix}_STF")


def stf_multipole_potential(moment: Tensor, name: str = "Phi") -> Tensor:
    """Потенциал STF-мультиполя: :math:`\\Phi = (-1)^\\ell / \\ell!\\; Q^{K} M^{(\\ell)}_{K}`.

    Параметры
    ----------
    moment : Tensor
        Симметричный бесследовый момент ранга ``l`` (обычно от
        :func:`stf_moment_symbols`).
    name : str
        Имя результата.

    Возвращает
    ----------
    Tensor
        Ранг 0 — скалярный потенциал как функция ``x, y, z, r``.

    Физический смысл
    ----------------
    Стандартное декартовое мультипольное разложение потенциала; при ``l = 1``
    даёт диполь ``(m . r)/r^3``, при ``l = 2`` — квадруполь
    ``Q_ij r_i r_j / r^5`` с точностью до нормировки.
    """
    order = moment.rank
    M = cartesian_stf_multipole(order)
    contraction = moment.full_inner(M)  # ранг 0 (работает и для order = 0)
    coeff = sp.Integer((-1) ** order) / sp.factorial(order)
    phi = sp.simplify(restore_r(coeff * contraction.as_scalar()))
    return Tensor.scalar(phi, name=name)


def stf_multipole_field(moment: Tensor, name: str = "g") -> Tensor:
    """Поле STF-мультиполя: :math:`\\mathbf g = -\\nabla\\Phi` (декартовы компоненты)."""
    phi = stf_multipole_potential(moment, name=f"{name}_phi")
    return -phi.grad(CARTESIAN_AXES, name=name).simplify(name=name)
