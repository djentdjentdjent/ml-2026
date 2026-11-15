"""Символьные координаты, метрика и базисы пакета ``mag_nav.symbolic``.

Здесь (и в :mod:`mag_nav.symbolic.core.conventions`) объявляются **все**
символьные переменные пакета. Остальные модули их только импортируют.

Содержание
----------
* Декартовы координаты ``x, y, z`` и радиус ``r``, сферические углы
  ``theta`` (полярный), ``phi`` (азимут); ``rho = sqrt(x^2 + y^2)``.
* Метрика евклидова: :math:`g_{ij} = \\delta_{ij}`, тензор Леви-Чивиты.
* Базисы: декартов ``(e_x, e_y, e_z)``; сферический ортонормированный
  ``(e_r, e_theta, e_phi)``; циклический (сферический) ``(e_{+1}, e_0, e_{-1})``.
* Переходы между координатными представлениями:
  :func:`spherical_to_cartesian`, :func:`cartesian_to_spherical`,
  :func:`eliminate_r`, :func:`restore_r`.
"""

from __future__ import annotations

import sympy as sp

from mag_nav.symbolic.core.conventions import I, PI, SQRT2
from mag_nav.symbolic.core.tensor import Tensor

__all__ = [
    "X", "Y", "Z", "R", "THETA", "PHI", "RHO",
    "CARTESIAN_COORDS", "SPHERICAL_COORDS",
    "E_X", "E_Y", "E_Z", "CARTESIAN_BASIS",
    "position_vector", "position_tensor", "r_squared", "r_expr",
    "metric", "inverse_metric", "kronecker_delta", "levi_civita",
    "unit_r", "unit_theta", "unit_phi", "spherical_frame_matrix",
    "frame_matrix", "cartesian_to_spherical_frame",
    "cyclic_basis_vector", "cyclic_basis", "cyclic_to_cartesian_matrix",
    "cartesian_to_cyclic_matrix", "to_cyclic_components", "to_cartesian_components",
    "spherical_subs", "angle_subs", "spherical_to_cartesian", "cartesian_to_spherical",
    "eliminate_r", "restore_r", "basis_tensor",
]

# ---------------------------------------------------------------------------
# Координатные символы
# ---------------------------------------------------------------------------

#: Декартовы координаты точки наблюдения.
X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
Z = sp.Symbol("z", real=True)

#: Радиус (расстояние от начала координат), положительный символ.
R = sp.Symbol("r", positive=True)

#: Полярный угол ``theta`` (``0 <= theta <= pi``, отсюда ``positive=True``:
#: ``sin(theta) >= 0`` и SymPy не порождает ``Abs(sin(theta))``) и азимут ``phi``.
THETA = sp.Symbol("theta", real=True, positive=True)
PHI = sp.Symbol("phi", real=True)

#: Цилиндрический радиус ``rho = sqrt(x^2 + y^2)`` (вспомогательное выражение).
RHO = sp.sqrt(X ** 2 + Y ** 2)

#: Кортеж декартовых координат.
CARTESIAN_COORDS = (X, Y, Z)

#: Кортеж сферических координат ``(r, theta, phi)``.
SPHERICAL_COORDS = (R, THETA, PHI)


def r_expr() -> sp.Expr:
    """:math:`r = \\sqrt{x^2 + y^2 + z^2}` как выражение от декартовых координат."""
    return sp.sqrt(X ** 2 + Y ** 2 + Z ** 2)


def r_squared() -> sp.Expr:
    """:math:`r^2 = x^2 + y^2 + z^2`."""
    return X ** 2 + Y ** 2 + Z ** 2


# ---------------------------------------------------------------------------
# Декартов базис
# ---------------------------------------------------------------------------

#: Орты декартова базиса как столбцы ``sympy.Matrix``.
E_X = sp.Matrix([1, 0, 0])
E_Y = sp.Matrix([0, 1, 0])
E_Z = sp.Matrix([0, 0, 1])

#: Именованный декартов базис.
CARTESIAN_BASIS = {"x": E_X, "y": E_Y, "z": E_Z}


def position_vector() -> sp.Matrix:
    """Вектор положения :math:`\\mathbf r = (x, y, z)^T` (``Matrix``)."""
    return sp.Matrix([X, Y, Z])


def position_tensor(name: str = "r") -> Tensor:
    """Вектор положения как :class:`Tensor` ранга 1."""
    return Tensor([X, Y, Z], name=name, indices=("i",))


# ---------------------------------------------------------------------------
# Метрика и псевдотензоры
# ---------------------------------------------------------------------------


def metric(dim: int = 3, name: str = "g") -> Tensor:
    """Евклидова метрика :math:`g_{ij} = \\delta_{ij}` (тензор ранга 2)."""
    return Tensor.delta(dim, name=name)


def inverse_metric(dim: int = 3, name: str = "g_inv") -> Tensor:
    """Обратная метрика :math:`g^{ij} = \\delta^{ij}` (в евклидовом базисе совпадает)."""
    return Tensor.delta(dim, name=name)


def kronecker_delta(dim: int = 3, name: str = "delta") -> Tensor:
    """Символ Кронекера :math:`\\delta_{ij}`."""
    return Tensor.delta(dim, name=name)


def levi_civita(dim: int = 3, name: str = "epsilon") -> Tensor:
    """Полностью антисимметричный тензор Леви-Чивиты :math:`\\varepsilon_{i_1...i_n}`."""
    return Tensor.levi_civita(dim, name=name)


# ---------------------------------------------------------------------------
# Сферический ортонормированный базис
# ---------------------------------------------------------------------------


def unit_r() -> sp.Matrix:
    """Единичный вектор :math:`\\hat{\\mathbf r}` в декартовых компонентах."""
    return sp.Matrix([sp.sin(THETA) * sp.cos(PHI), sp.sin(THETA) * sp.sin(PHI), sp.cos(THETA)])


def unit_theta() -> sp.Matrix:
    """Единичный вектор :math:`\\hat{\\mathbf e}_\\theta` (в сторону роста ``theta``)."""
    return sp.Matrix([sp.cos(THETA) * sp.cos(PHI), sp.cos(THETA) * sp.sin(PHI), -sp.sin(THETA)])


def unit_phi() -> sp.Matrix:
    """Единичный вектор :math:`\\hat{\\mathbf e}_\\varphi` (в сторону роста ``phi``)."""
    return sp.Matrix([-sp.sin(PHI), sp.cos(PHI), sp.Integer(0)])


def spherical_frame_matrix() -> sp.Matrix:
    """Матрица :math:`3\\times3` со столбцами :math:`(\\hat e_r, \\hat e_\\theta, \\hat e_\\varphi)`.

    Физические компоненты вектора в сферическом базисе ``v_sph`` связаны с
    декартовыми ``v_cart`` как ``v_cart = spherical_frame_matrix() @ v_sph``.
    Матрица ортогональна: ``M.T @ M = I``.
    """
    return sp.Matrix.hstack(unit_r(), unit_theta(), unit_phi())


#: Синоним :func:`spherical_frame_matrix`.
frame_matrix = spherical_frame_matrix


def cartesian_to_spherical_frame() -> sp.Matrix:
    """Обратное преобразование базиса: ``v_sph = M.T @ v_cart`` (матрица ортогональна)."""
    return spherical_frame_matrix().T


# ---------------------------------------------------------------------------
# Циклический (сферический) базис: e_{+1}, e_0, e_{-1}
# ---------------------------------------------------------------------------

_CYCLIC_CONVENTION = (
    "e_{+1} = -(e_x + I e_y)/sqrt(2),  e_0 = e_z,  e_{-1} = (e_x - I e_y)/sqrt(2)"
)


def cyclic_basis_vector(s: int) -> sp.Matrix:
    """Вектор циклического базиса ``e_s`` (``s`` в ``{+1, 0, -1}``) в декартовых компонентах.

    Параметры
    ---------
    s : int
        Проекция момента ``+1``, ``0`` или ``-1``.

    Возвращает
    ----------
    sympy.Matrix
        Столбец из трёх (вообще комплексных) компонент.

    Соглашение
    -----------
    :math:`e_{+1} = -(e_x + i e_y)/\\sqrt{2}`, :math:`e_0 = e_z`,
    :math:`e_{-1} = (e_x - i e_y)/\\sqrt2` — стандартное (Condon–Shortley),
    согласованное с :math:`Y_{1,\\pm1} = \\mp\\sqrt{3/(8\\pi)}\\sin\\theta\\,e^{\\pm i\\varphi}`.
    """
    s = int(s)
    if s == 1:
        return -(E_X + I * E_Y) / SQRT2
    if s == 0:
        return E_Z
    if s == -1:
        return (E_X - I * E_Y) / SQRT2
    raise ValueError(f"cyclic_basis_vector: s должен быть в (+1, 0, -1), получено {s}")


def cyclic_basis() -> dict:
    """Словарь ``{+1: e_{+1}, 0: e_0, -1: e_{-1}}`` (``Matrix``-столбцы)."""
    return {s: cyclic_basis_vector(s) for s in (1, 0, -1)}


def cyclic_to_cartesian_matrix() -> sp.Matrix:
    """Матрица перехода ``v_cart = S @ v_cyc`` для циклического базиса (столбцы — ``e_s``)."""
    return sp.Matrix.hstack(cyclic_basis_vector(1), cyclic_basis_vector(0), cyclic_basis_vector(-1))


def cartesian_to_cyclic_matrix() -> sp.Matrix:
    """Обратный переход ``v_cyc = S^{-1} @ v_cart`` (унитарная матрица: ``S^{-1} = S^H``)."""
    return cyclic_to_cartesian_matrix().adjoint()


def to_cyclic_components(v_cart) -> sp.Matrix:
    """Компоненты вектора в циклическом базисе ``(v_{+1}, v_0, v_{-1})``."""
    v = sp.Matrix(v_cart) if not isinstance(v_cart, sp.MatrixBase) else v_cart
    return cartesian_to_cyclic_matrix() @ v


def to_cartesian_components(v_cyc) -> sp.Matrix:
    """Обратно: декартовы компоненты по циклическим ``(v_{+1}, v_0, v_{-1})``."""
    v = sp.Matrix(v_cyc) if not isinstance(v_cyc, sp.MatrixBase) else v_cyc
    return cyclic_to_cartesian_matrix() @ v


# ---------------------------------------------------------------------------
# Переходы между координатными представлениями
# ---------------------------------------------------------------------------


def spherical_subs() -> dict:
    """Подстановка ``x, y, z -> (r, theta, phi)``: декартовы -> сферические."""
    return {
        X: R * sp.sin(THETA) * sp.cos(PHI),
        Y: R * sp.sin(THETA) * sp.sin(PHI),
        Z: R * sp.cos(THETA),
    }


def angle_subs() -> dict:
    """Подстановка углов через декартовы координаты: ``cos(theta) -> z/r``, ``sin(theta) -> rho/r``."""
    return {
        sp.cos(THETA): Z / R,
        sp.sin(THETA): RHO / R,
    }


#: Функции от ``theta``, которые нужно уметь переводить в ``x, y, z, r``.
def _trig_of_theta_subs() -> dict:
    """Все тригонометрические функции от ``theta`` через ``x, y, z, r``."""
    pairs = [
        (sp.sin(THETA), RHO / R),
        (sp.cos(THETA), Z / R),
        (sp.tan(THETA), RHO / Z),
    ]
    for fname, repl in (("cot", Z / RHO), ("sec", R / RHO), ("csc", R / RHO)):
        func = getattr(sp, fname, None)
        if func is not None:
            pairs.append((func(THETA), repl))
    return dict(pairs)


def _azimuth_subs(expr: sp.Expr) -> sp.Expr:
    """Замена азимутальных функций: ``exp(I*k*phi) -> ((x + I y)/rho)^k``, ``cos/sin(phi)``."""

    def _replace(ex: sp.Expr) -> sp.Expr:
        if isinstance(ex, sp.exp):
            arg = ex.args[0]
            if arg.has(PHI):
                k = sp.simplify(arg / (I * PHI))
                if k.is_integer:
                    return ((X + I * Y) / RHO) ** int(k)
            return ex
        if isinstance(ex, sp.cos) and ex.args == (PHI,):
            return X / RHO
        if isinstance(ex, sp.sin) and ex.args == (PHI,):
            return Y / RHO
        return ex

    return expr.replace(lambda ex: isinstance(ex, (sp.exp, sp.cos, sp.sin)), _replace)


def spherical_to_cartesian(expr, keep_r: bool = True, do_simplify: bool = True) -> sp.Expr:
    """Переписывает выражение из ``(r, theta, phi)`` в ``(x, y, z[, r])``.

    Алгоритм: ``expand_trig`` (чтобы ``sin(2 theta)`` и т.п. разбились на
    степени простых ``sin/cos``), затем замена тригонометрии полярного угла,
    замена ``exp(I*m*phi)``, затем ``powdenest`` (чтобы ``sqrt(30 x^2 + 30 y^2)``
    сократилось с ``rho``).

    Параметры
    ---------
    expr : sympy выражение
        Выражение от ``r, theta, phi``.
    keep_r : bool
        Если ``True`` (по умолчанию), ``r`` остаётся символом (считается
        ``r = sqrt(x^2 + y^2 + z^2)``); если ``False``, ``r`` устраняется
        функцией :func:`eliminate_r`.
    do_simplify : bool
        Применить ``sp.simplify`` к результату.

    Возвращает
    ----------
    sympy.Expr
        Выражение только от ``x, y, z`` (и, опционально, ``r``).
    """
    e = sp.sympify(expr)
    e = sp.expand_trig(sp.simplify(e))
    e = e.subs(_trig_of_theta_subs())
    e = _azimuth_subs(e)
    e = _trig_of_theta_subs_residue(e)
    e = sp.powdenest(e, force=True)
    e = sp.simplify(e) if do_simplify else e
    if not keep_r:
        e = eliminate_r(e)
    return e


def _trig_of_theta_subs_residue(e: sp.Expr) -> sp.Expr:
    """Добивает оставшиеся ``sin/cos`` от ``theta`` (например, ``sin(theta)**(-1)``)."""
    if not (e.has(sp.sin, THETA) or e.has(sp.cos, THETA)):
        return e

    def _rep(ex):
        if isinstance(ex, sp.sin) and ex.args == (THETA,):
            return RHO / R
        if isinstance(ex, sp.cos) and ex.args == (THETA,):
            return Z / R
        return ex

    return e.replace(lambda ex: isinstance(ex, (sp.sin, sp.cos)), _rep)


def cartesian_to_spherical(expr, do_simplify: bool = True) -> sp.Expr:
    """Переписывает выражение из ``(x, y, z)`` в ``(r, theta, phi)``.

    Подстановка ``x = r sin(theta)cos(phi)`` и т.д. с последующим упрощением.
    """
    e = sp.sympify(expr).subs(spherical_subs())
    return sp.simplify(e) if do_simplify else e


def eliminate_r(expr, do_simplify: bool = True) -> sp.Expr:
    """Устраняет символ ``r`` заменой ``r -> sqrt(x^2 + y^2 + z^2)``."""
    e = sp.sympify(expr).subs(R, r_expr())
    e = sp.powdenest(e, force=True)
    return sp.simplify(e) if do_simplify else e


def restore_r(expr, do_simplify: bool = True) -> sp.Expr:
    """Сжатие выражений: ``x^2 + y^2 + z^2 -> r^2`` (и любая степень --- к ``r``).

    Нужна после дифференцирования по декартовым координатам: там ``r`` возникает
    как ``(x^2 + y^2 + z^2)**(-3/2)``, а не как явный ``sqrt``, поэтому ordinary
    ``subs(sqrt(...), r)`` не сработает — меняем именно основание степени.
    """
    e = sp.sympify(expr)
    s = X ** 2 + Y ** 2 + Z ** 2

    def _replace(ex: sp.Expr) -> sp.Expr:
        if isinstance(ex, sp.Pow) and ex.args[0] == s:
            return R ** (2 * ex.args[1])
        return ex

    e = e.replace(lambda ex: isinstance(ex, sp.Pow) and ex.args[0] == s, _replace)
    e = e.subs({s: R ** 2, r_expr(): R})
    return sp.simplify(e) if do_simplify else e


# ---------------------------------------------------------------------------
# Универсальный приводитель базисных векторов
# ---------------------------------------------------------------------------


def basis_tensor(obj, name: str = "e") -> Tensor:
    """Приводит векторную величину любого вида к :class:`Tensor` ранга 1.

    Принимает :class:`Tensor`, ``sympy.Matrix`` (столбец/строку), список или
    кортеж компонент. Скаляр приводится к тензору ранга 0.
    """
    if isinstance(obj, Tensor):
        if obj.rank not in (0, 1):
            raise ValueError(f"basis_tensor: ожидается скаляр или вектор, получен ранг {obj.rank}")
        return obj
    if isinstance(obj, sp.MatrixBase):
        return Tensor.from_matrix(obj, name=name)
    if isinstance(obj, (list, tuple)):
        return Tensor(list(obj), name=name)
    if isinstance(obj, (sp.Expr, int, float, sp.Number)):
        return Tensor.scalar(obj, name=name)
    raise TypeError(f"basis_tensor: неподдерживаемый тип {type(obj)}")


def conventions_note() -> str:
    """Короткая справка по базисным соглашениям (для логов и отчётов)."""
    return (
        f"cyclic basis: {_CYCLIC_CONVENTION}; "
        "spherical frame: (e_r, e_theta, e_phi), v_cart = M @ v_phys; "
        "metric: g_ij = delta_ij"
    )
