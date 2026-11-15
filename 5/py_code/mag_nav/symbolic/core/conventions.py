"""Фазовые соглашения и физические константы пакета ``mag_nav.symbolic``.

Этот модуль — **единственный источник истины** про знаки, фазы, нормировки
и обозначения. Внешний код не диктует соглашений этому пакету; наоборот,
все остальные модули импортируют символы и конвенции отсюда и из
:mod:`mag_nav.symbolic.core.coords`.

Зафиксированные соглашения (см. также словарь :data:`CONVENTIONS`)
-----------------------------------------------------------------
* Сферические гармоники: ортонормированные, физическая конвенция,
  ``Y_{lm}(theta, phi) = sqrt((2l+1)/(4 pi) * (l-m)!/(l+m)!) * P_l^m(cos theta) * exp(i m phi)``,
  фаза Кондона — Шортли включена (``Y_1^1 = -sqrt(3/(8 pi)) sin theta e^{i phi}``).
  Реализация — ``sympy.Ynm(l, m, theta, phi)``; ``theta`` — полярный угол,
  ``phi`` — азимут.
* Временáя зависимость: ``exp(+I omega t)``. Физическое поле
  ``B(r, t) = Re[c(r) exp(I omega t)]``, где ``c(r)`` — комплексная амплитуда.
* Комплексная векторная величина: ``c = a - I b``, где ``a = Re c``, ``b = -Im c``
  — то есть ``a`` и ``b`` есть две вещественные «квадратуры» поля; знак при ``I``
  фиксирован здесь и нигде более.
* Комплексный магнитный момент вращающегося диполя:
  ``m = m0 * (e1 - I e2)``, где ``e1``, ``e2`` — орты плоскости вращения
  (порядок пары ``(e1, e2)`` задаёт направление вращения, handedness).
* Демодуляция / правило отбора: множитель ``exp(-I m omega t)``, после усреднения
  по периоду выживает ``m = -sign(omega)``.
* Единицы: СИ; ``mu_0/(4 pi)`` — единый символ :data:`MU0_OVER_4PI`.
"""

from __future__ import annotations

import sympy as sp

__all__ = [
    "I",
    "PI",
    "SQRT2",
    "OMEGA",
    "TIME",
    "PERIOD",
    "MU0",
    "MU0_OVER_4PI",
    "FIELD_AMPLITUDE",
    "CONVENTIONS",
    "time_phase",
    "demod_phase",
    "real_signal",
    "complex_pair",
    "complex_moment",
    "demod_selected_m",
    "validate_lm",
    "is_concrete_integer",
    "is_concrete",
]

# ---------------------------------------------------------------------------
# Базовые математические константы
# ---------------------------------------------------------------------------

#: Комплексная единица.
I = sp.I

#: Число pi.
PI = sp.pi

#: sqrt(2) — встречается в переходах между базисами.
SQRT2 = sp.sqrt(2)

# ---------------------------------------------------------------------------
# Символы (создаются ТОЛЬКО здесь и в coords.py)
# ---------------------------------------------------------------------------

#: Круговая частота вращения маяка, ``omega`` (вещественная, знак важен).
OMEGA = sp.Symbol("omega", real=True)

#: Время, ``t``.
TIME = sp.Symbol("t", real=True)

#: Период вращения, ``T`` (положительный).
PERIOD = sp.Symbol("T", positive=True)

#: Магнитная постоянная ``mu_0``.
MU0 = sp.Symbol("mu_0", positive=True)

#: Константа би-о-савара ``mu_0 / (4 pi)`` — единый символ во всех формулах.
MU0_OVER_4PI = MU0 / (4 * PI)

#: Скалярная комплексная амплитуда потенциала ``c(r)`` (для однокомпонентных задач).
FIELD_AMPLITUDE = sp.Symbol("c", complex=True)


# ---------------------------------------------------------------------------
# Реестр соглашений (для печати в отчётах и для себя)
# ---------------------------------------------------------------------------

CONVENTIONS: dict[str, str] = {
    "harmonics": "orthonormal complex Y_{lm}, physics convention, Condon-Shortley phase ON",
    "harmonic_arguments": "Ynm(l, m, theta, phi): theta - polar, phi - azimuthal",
    "time_dependence": "exp(+I*omega*t)",
    "physical_field": "B(r,t) = Re[ c(r) * exp(+I*omega*t) ]",
    "complex_vector": "c = a - I*b  (a = Re c, b = -Im c)",
    "complex_moment": "m = m0 * (e1 - I*e2)",
    "demodulation": "factor exp(-I*m*omega*t); surviving order m = -sign(omega)",
    "units": "SI, mu_0/(4*pi) as a single symbol",
    "metric": "Euclidean, g_ij = delta_ij in Cartesian orthonormal frame",
}


def describe_conventions() -> str:
    """Возвращает человекочитаемое описание всех зафиксированных соглашений.

    Returns
    -------
    str
        Многострочный текст ``ключ = значение``.
    """
    return "\n".join(f"  {k} = {v}" for k, v in sorted(CONVENTIONS.items()))


# ---------------------------------------------------------------------------
# Фазовые множители
# ---------------------------------------------------------------------------


def time_phase(omega: sp.Basic = OMEGA, t: sp.Basic = TIME) -> sp.Expr:
    """Временнóй фазовый множитель ``exp(+I*omega*t)``.

    Параметры
    ---------
    omega : sympy выражение
        Круговая частота (по умолчанию :data:`OMEGA`).
    t : sympy выражение
        Время (по умолчанию :data:`TIME`).

    Возвращает
    ----------
    sympy.Expr
        ``exp(I*omega*t)``.

    Физический смысл
    ----------------
    Комплексная амплитуда ``c(r)`` умножается на этот множитель, затем
    берётся действительная часть: :func:`real_signal`.
    """
    return sp.exp(I * omega * t)


def demod_phase(m: int, omega: sp.Basic = OMEGA, t: sp.Basic = TIME) -> sp.Expr:
    """Множитель демодуляции гармоники порядка ``m``: ``exp(-I*m*omega*t)``.

    Параметры
    ---------
    m : int
        Азимутальный индекс мультиполя.
    omega, t : sympy выражение
        Частота и время.

    Возвращает
    ----------
    sympy.Expr
        ``exp(-I*m*omega*t)``.

    Физический смысл
    ----------------
    Вращение источника с угловой скоростью ``omega`` превращает
    ``q_{lm}(t) = exp(-I*m*omega*t) * q~_{lm}``. Умножение сигнала на этот
    множитель и усреднение по периоду выделяет единственный индекс
    ``m = -sign(omega)`` (см. :func:`demod_selected_m`).
    """
    return sp.exp(-I * sp.Integer(m) * omega * t)


def real_signal(c_expr: sp.Basic, omega: sp.Basic = OMEGA, t: sp.Basic = TIME) -> sp.Expr:
    """Физическая (вещественная) форма комплексного сигнала.

    Параметры
    ---------
    c_expr : sympy выражение
        Комплексная амплитуда ``c(r)``.
    omega, t : sympy выражение
        Частота и время.

    Возвращает
    ----------
    sympy.Expr
        ``Re[ c_expr * exp(I*omega*t) ]`` (по возможности разложенное).
    """
    return sp.re(sp.expand_complex(c_expr * time_phase(omega, t)))


def complex_pair(a: sp.Basic, b: sp.Basic) -> sp.Expr:
    """Комплексная величина из двух квадратур: ``c = a - I*b``.

    Параметры
    ---------
    a, b : sympy выражения
        Первая и вторая квадратуры (вещественные поля).

    Возвращает
    ----------
    sympy.Expr
        ``a - I*b``.

    Физический смысл
    ----------------
    Знак при ``I`` зафиксирован соглашением пакета: если ``a`` — поле в момент
    ``t = 0``, а ``b`` — поле в момент четверти периода, то ``c = a - I*b``
    есть комплексная амплитуда при конвенции ``exp(+I*omega*t)``.
    """
    return sp.sympify(a) - I * sp.sympify(b)


def complex_moment(m0: sp.Basic, e1=None, e2=None):
    """Комплексный момент вращающегося диполя: ``m = m0*(e1 - I*e2)``.

    Параметры
    ----------
    m0 : sympy выражение
        Модуль момента.
    e1, e2
        Орты плоскости вращения. Могут быть переданы как ``Tensor`` ранга 1,
        как ``sympy.Matrix`` (столбец) или как кортежи компонент; по умолчанию
        берутся декартовы базисные векторы ``x`` и ``y``.

    Возвращает
    ----------
    тот же тип, что и ``e1``
        ``m0*(e1 - I*e2)``.

    Физический смысл
    ----------------
    Момент ``m(t) = m0*(e1*cos(omega*t) + e2*sin(omega*t))`` — вращающийся в
    плоскости ``(e1, e2)`` диполь — имеет комплексную амплитуду
    ``m = m0*(e1 - I*e2)`` при конвенции ``Re[m e^{I omega t}]``.
    """
    # Локальный импорт, чтобы не создавать цикл coords -> conventions -> coords.
    from mag_nav.symbolic.core.coords import E_X, E_Y, basis_tensor

    if e1 is None:
        e1 = E_X
        e2 = E_Y
    m0 = sp.sympify(m0)
    if e2 is None:
        raise ValueError("complex_moment: нужно задать оба орта (e1, e2)")
    # Приводим произвольные контейнеры к Tensor ранга 1
    t1 = basis_tensor(e1)
    t2 = basis_tensor(e2)
    return m0 * (t1 - I * t2)


def demod_selected_m(omega: sp.Basic = OMEGA) -> sp.Expr:
    """Выживающий азимутальный индекс после демодуляции: ``m* = -sign(omega)``.

    Параметры
    ----------
    omega : sympy выражение
        Круговая частота.

    Возвращает
    ----------
    sympy.Expr
        ``-sign(omega)`` (для конкретной вещественной ``omega`` даёт ``±1``).
    """
    return -sp.sign(omega)


# ---------------------------------------------------------------------------
# Валидация индексов
# ---------------------------------------------------------------------------


def is_concrete(value: sp.Basic) -> bool:
    """``True``, если выражение не содержит свободных символов."""
    return len(sp.sympify(value).free_symbols) == 0


def is_concrete_integer(value: sp.Basic) -> bool:
    """``True``, если выражение — конкретное целое (Integer/Rational с знаменателем 1)."""
    v = sp.sympify(value)
    if not is_concrete(v):
        return False
    if v.is_Integer:
        return True
    return v.is_rational and v == int(v)


def validate_lm(l: sp.Basic, m: sp.Basic, allow_symbolic_l: bool = True) -> None:
    """Проверяет допустимость пары ``(l, m)``.

    Параметры
    ----------
    l, m : sympy выражения
        Порядок мультиполя и азимутальный индекс.
    allow_symbolic_l : bool
        Если ``True`` (по умолчанию), ``l`` может оставаться символьным —
        тогда проверка ``|m| <= l`` выполняется только когда ``l`` конкретен.

    Возвращает
    ----------
    None

    Raises
    ------
    ValueError
        Если ``m`` конкретен и нецелый, либо ``l`` конкретен и нарушено
        ``|m| <= l`` либо ``l < 0``.
    """
    l = sp.sympify(l)
    m = sp.sympify(m)

    if is_concrete(m):
        if not m.is_integer:
            raise ValueError(f"m должен быть целым, получено {m}")
    if is_concrete(l):
        if not l.is_integer:
            raise ValueError(f"l должен быть целым, получено {l}")
        if l < 0:
            raise ValueError(f"l должен быть >= 0, получено {l}")
        if is_concrete(m) and abs(int(m)) > int(l):
            raise ValueError(f"требуется |m| <= l, получено l={l}, m={m}")
    elif not allow_symbolic_l:
        raise ValueError("порядок l должен быть конкретным целым")
