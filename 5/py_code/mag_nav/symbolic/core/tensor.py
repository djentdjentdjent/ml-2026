"""Тензорный объект пакета ``mag_nav.symbolic``.

Базовая сущность — :class:`Tensor`: обёртка над ``sympy.ImmutableDenseNDimArray``
с метаданными (ранг, форма, симметрия, имя). Компоненты — внутренняя реализация,
доступная по запросу; операции (внешнее произведение, свёртка, симметризация,
снятие следа, STF-проекция) возвращают **новый** :class:`Tensor` с корректным
рангом и симметрией, а не россыпь скаляров.

Метрика по умолчанию — евклидова :math:`g_{ij} = \\delta_{ij}` (ортонормированный
декартов базис), поэтому подъём/опускание индексов тождественны, если явно не
передан метрический тензор.
"""

from __future__ import annotations

import functools
import itertools
from typing import Iterator, Sequence

import sympy as sp
from sympy.tensor.array import ImmutableDenseNDimArray, MutableDenseNDimArray

__all__ = ["Tensor", "tensor_from", "stf_coefficients",
           "stf_coefficients_closed_form", "stf_coefficients_derived"]


# ---------------------------------------------------------------------------
# Вспомогательные функции над NDimArray
# ---------------------------------------------------------------------------


def _as_ndim_array(data) -> ImmutableDenseNDimArray:
    """Приводит вход к ``ImmutableDenseNDimArray`` (скаляр оформляется рангом 0)."""
    if isinstance(data, Tensor):
        return data.data
    if isinstance(data, ImmutableDenseNDimArray):
        return data
    if isinstance(data, MutableDenseNDimArray):
        return data.as_immutable()
    if isinstance(data, sp.MatrixBase):
        if data.rows == 1 or data.cols == 1:
            return ImmutableDenseNDimArray([sp.sympify(v) for v in data])
        return ImmutableDenseNDimArray([[sp.sympify(v) for v in row] for row in data.tolist()])
    expr = sp.sympify(data)
    return ImmutableDenseNDimArray(expr)


def _wrap(result) -> ImmutableDenseNDimArray:
    """Результат sympy-операции над массивом: скаляр превращаем в массив ранга 0."""
    if isinstance(result, (ImmutableDenseNDimArray, MutableDenseNDimArray)):
        return result.as_immutable()
    return ImmutableDenseNDimArray(sp.sympify(result))


def _flatten(arr: ImmutableDenseNDimArray) -> list:
    """Список всех компонент (для ранга 0 — один элемент).

    ``list(arr)`` у ``NDimArray`` перебирает только верхний уровень, поэтому
    обходим структуру через ``tolist()`` рекурсивно.
    """
    if arr.rank() == 0:
        return [sp.sympify(arr[()])]
    out: list = []

    def walk(obj) -> None:
        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        else:
            out.append(sp.sympify(obj))

    walk(arr.tolist())
    return out


def _iter_indices(shape: Sequence[int]) -> Iterator[tuple[int, ...]]:
    return itertools.product(*[range(s) for s in shape])


def _permute_array(arr: ImmutableDenseNDimArray, order: Sequence[int]) -> ImmutableDenseNDimArray:
    """Перестановка осей: ``new[k_0..k_{n-1}] = arr[s_0..s_{n-1}]``, где ``s[order[j]] = k_j``.

    Иными словами, ось ``j`` результата есть ось ``order[j]`` исходного массива.
    """
    order = tuple(int(o) for o in order)
    n = arr.rank()
    if sorted(order) != list(range(n)):
        raise ValueError(f"order {order} не является перестановкой range({n})")
    if n == 0:
        return arr
    shape = tuple(arr.shape[order[j]] for j in range(n))
    out = MutableDenseNDimArray.zeros(*shape)
    for idx in _iter_indices(shape):
        src = [0] * n
        for j, k in enumerate(idx):
            src[order[j]] = k
        out[idx] = arr[tuple(src)]
    return out.as_immutable()


def _permutation_sign(perm: Sequence[int]) -> int:
    """Знак перестановки (по числу инверсий)."""
    inv = 0
    for a in range(len(perm)):
        for b in range(a + 1, len(perm)):
            if perm[a] > perm[b]:
                inv += 1
    return -1 if inv % 2 else 1


def _symmetrize_axes(arr: ImmutableDenseNDimArray, axes: Sequence[int]) -> ImmutableDenseNDimArray:
    """Среднее по всем перестановкам указанных осей (вес ``1/k!``)."""
    axes = tuple(int(a) for a in axes)
    n = arr.rank()
    acc = None
    count = 0
    for perm in itertools.permutations(axes):
        order = list(range(n))
        for target, source in zip(axes, perm):
            order[target] = source
        p = _permute_array(arr, order)
        acc = p if acc is None else acc + p
        count += 1
    return _wrap(sp.sympify(acc) / count)


def _antisymmetrize_axes(arr: ImmutableDenseNDimArray, axes: Sequence[int]) -> ImmutableDenseNDimArray:
    """Альтернированное среднее по перестановкам указанных осей (вес ``sign/k!``)."""
    axes = tuple(int(a) for a in axes)
    n = arr.rank()
    acc = None
    count = 0
    for perm in itertools.permutations(axes):
        order = list(range(n))
        for target, source in zip(axes, perm):
            order[target] = source
        p = _permutation_sign(perm) * _permute_array(arr, order)
        acc = p if acc is None else acc + p
        count += 1
    return _wrap(sp.sympify(acc) / count)


def _symmetric_basis_tensors(rank: int, dim: int) -> list:
    """Базис симметричных тензоров ранга ``rank`` (по одному на независимый компонент).

    Используется для точного вывода коэффициентов STF-проекции: условие
    «след нулевой для любого симметричного тензора» проверяется на этом базисе,
    что даёт числовую (рациональную) линейную систему вместо символической.
    """
    rank = int(rank)
    dim = int(dim)
    independent = sorted({tuple(sorted(t)) for t in _iter_indices((dim,) * rank)})
    basis = []
    for ss in independent:
        base = MutableDenseNDimArray.zeros(*((dim,) * rank))
        for t in _iter_indices((dim,) * rank):
            if tuple(sorted(t)) == ss:
                base[t] = sp.Integer(1)
        basis.append(base.as_immutable())
    return basis


def stf_coefficients_derived(rank: int, dim: int) -> tuple:
    """Коэффициенты STF-проекции, полученные точным линейно-алгебраическим выводом.

    Ищем :math:`Q = \\sum_{s=0}^{\\lfloor k/2\\rfloor} c_s\\,
    \\mathrm{sym}(\\delta^{\\otimes s} \\otimes \\mathrm{Tr}^{(s)} T)` с
    :math:`c_0 = 1`, такой что однократный след ``Q`` тождественно равен нулю по
    любому симметричному тензору ``T``. Условия линейны по ``c_s``, поэтому
    коэффициенты находятся как одномерное ядро матрицы следов: никаких формул
    «из памяти», всё выводится символически и затем проверяется тестами.

    Параметры
    ----------
    rank : int
        Ранг симметричного тензора (``rank >= 0``).
    dim : int
        Размерность пространства (обычно 3).

    Возвращает
    ----------
    tuple[sympy.Rational, ...]
        ``(c_0, c_1, ..., c_floor(rank/2))``, где ``c_0 == 1``.
    """
    rank = int(rank)
    dim = int(dim)
    n_pairs = rank // 2
    if n_pairs == 0:
        return (sp.Integer(1),)

    delta = ImmutableDenseNDimArray(sp.eye(dim).tolist())

    def delta_trace_terms(T):
        """Слагаемые :math:`\\mathrm{sym}(\\delta^{\\otimes s} \\otimes \\mathrm{Tr}^{(s)}T)`, ``s = 0..``."""
        out = [T]
        traced = T
        for s in range(1, n_pairs + 1):
            # s-кратный след: каждый раз свёртка первых двух осей (T симметричен).
            traced = _wrap(sp.tensorcontraction(traced, [0, 1]))
            emb = traced
            for _ in range(s):
                emb = _wrap(sp.tensorproduct(delta, emb))
            out.append(_symmetrize_axes(emb, tuple(range(rank))))
        return out

    # Условие «след нулевой для ЛЮБОГО симметричного T» линейно по c_s, поэтому
    # достаточно потребовать его на базисных симметричных тензорах. Решаем
    # инкрементально: строки добавляются по одному базисному тензору, пока ядро
    # не станет одномерным (обычно хватает нескольких). Это точная рациональная
    # линейная алгебра — без «формл из памяти».
    matrix_rows: list = []
    vec = None
    for T in _symmetric_basis_tensors(rank, dim):
        # компоненты уже точные рациональные: nsimplify здесь не нужен (и очень дорог)
        traces = [[sp.sympify(cc) for cc in _flatten(_wrap(sp.tensorcontraction(term, [0, 1])))]
                  for term in delta_trace_terms(T)]
        n_comp = len(traces[0])
        for alpha in range(n_comp):
            matrix_rows.append([traces[s][alpha] for s in range(len(traces))])
        ns = sp.Matrix(matrix_rows).nullspace()
        if len(ns) == 1:
            candidate = ns[0].applyfunc(sp.simplify)
            if candidate[0] != 0:
                vec = candidate / candidate[0]
                break
    if vec is None:
        ns = sp.Matrix(matrix_rows).nullspace() if matrix_rows else []
        raise RuntimeError(
            f"STF-коэффициенты для rank={rank}, dim={dim}: ядро не одномерно "
            f"(размерность {len(ns)}), единственность не установлена"
        )
    coeffs = tuple(sp.nsimplify(sp.simplify(v)) for v in vec)

    # Контроль: на найденных коэффициентах след обязан исчезать на СЛУЧАЙНЫХ
    # целочисленных симметричных тензорах (быстрая проверка без полной базы).
    import random as _random

    for seed in range(3):
        rng = _random.Random(seed * 977 + rank * 31 + dim)
        for _ in range(2):
            comps = [sp.Integer(rng.randint(1, 5)) for _ in range(dim ** rank)]
            T_rand = sp.MutableDenseNDimArray(comps).reshape(*((dim,) * rank)).as_immutable()
            terms = delta_trace_terms(_symmetrize_axes(T_rand, tuple(range(rank))))
            Q = terms[0] * coeffs[0]
            for coef, term in zip(coeffs[1:], terms[1:]):
                Q = Q + term * coef
            tr = _flatten(_wrap(sp.tensorcontraction(_wrap(Q), [0, 1])))
            if any(t != 0 for t in tr):
                raise RuntimeError(
                    f"STF-коэффициенты для rank={rank}, dim={dim}: контроль не пройден"
                )
    return coeffs


def stf_coefficients_closed_form(rank: int, dim: int) -> tuple:
    """Аналитические коэффициенты STF-проекции (быстрые, любой ранг).

    :math:`c_0 = 1`, :math:`c_s = (-1)^s (2s-1)!! \binom{k}{2s}
    / \prod_{j=1}^{s}(2k + n - 2 - 2j)` для симметричного тензора ранга
    :math:`k` в :math:`n`-мерном евклидовом пространстве, где симметризация
    нормирована (сумма по всем перестановкам делить на ``k!``).

    Формула не «взята из справочника», а подогнана и проверена по значениям
    :func:`stf_coefficients_derived` (см. :func:`stf_coefficients` и тесты).

    Параметры
    ----------
    rank : int
        Ранг симметричного тензора.
    dim : int
        Размерность пространства.

    Возвращает
    ----------
    tuple[sympy.Rational, ...]
        ``(c_0, ..., c_floor(rank/2))``.
    """
    rank = int(rank)
    dim = int(dim)
    coeffs = [sp.Integer(1)]
    for s in range(1, rank // 2 + 1):
        numerator = int(sp.factorial2(2 * s - 1)) * int(sp.binomial(rank, 2 * s))
        denominator = 1
        for j in range(1, s + 1):
            factor = 2 * rank + dim - 2 - 2 * j
            if factor <= 0:
                raise ValueError(
                    f"STF: вырожденный знаменатель при rank={rank}, dim={dim}"
                )
            denominator *= factor
        value = sp.Rational(numerator, denominator)
        coeffs.append(-value if s % 2 else value)
    return tuple(coeffs)


@functools.lru_cache(maxsize=None)
def stf_coefficients(rank: int, dim: int) -> tuple:
    """Коэффициенты STF-проекции: закрытая формула, сверенная с выводом.

    Для ``rank <= 3`` значения дополнительно сверяются с точной линейной
    системой :func:`stf_coefficients_derived` (это дёшево); при расхождении
    бросается ``RuntimeError``. Для больших рангов используется закрытая форма.
    """
    rank = int(rank)
    dim = int(dim)
    coeffs = stf_coefficients_closed_form(rank, dim)
    if rank <= 3:
        derived = stf_coefficients_derived(rank, dim)
        if tuple(sp.nsimplify(c) for c in coeffs) != tuple(sp.nsimplify(c) for c in derived):
            raise RuntimeError(
                f"STF-коэффициенты rank={rank}, dim={dim}: закрытая формула "
                f"{coeffs} не совпала с выводом {derived}"
            )
    return coeffs


# ---------------------------------------------------------------------------
# Класс Tensor
# ---------------------------------------------------------------------------


class Tensor:
    """Символьный тензор с метаданными.

    Атрибуты
    --------
    rank : int
        Число индексов (ранг). ``0`` — скаляр.
    shape : tuple[int, ...]
        Размерности по каждому индексу.
    symmetry : str | None
        Метка симметрии: ``'symmetric'``, ``'antisymmetric'``, ``'traceless'``,
        ``'stf'`` (симметричный + бесследовый) или ``None``. Метка — это
        гарантия, выставленная конструктором/операцией; она не проверяется
        «на лету» при каждом вызове (для проверки есть :meth:`is_symmetric`
        и т.п.).
    name : str
        Имя для отображения (например ``'B'``, ``'Jacobian'``).
    indices : tuple[str, ...] | None
        Имена индексов, например ``('i', 'j')`` — полезно при печати формул.
    data : sympy.ImmutableDenseNDimArray
        Компоненты. Доступ к «сырым» компонентам — :attr:`data` или
        :meth:`components`.

    Пример
    ------
    >>> from mag_nav.symbolic.core.tensor import Tensor
    >>> import sympy as sp
    >>> a, b = sp.symbols('a b')
    >>> T = Tensor([[a, b], [-b, a]], name='M')
    >>> T.rank, T.shape
    (2, (2, 2))
    >>> T.traceless(0, 1).trace().equals(0)
    True
    """

    __slots__ = ("_data", "_name", "_symmetry", "_indices")

    # ------------------------------------------------------------------ init

    def __init__(self, data, name: str = "T", symmetry: str | None = None,
                 indices: Sequence[str] | None = None, rank: int | None = None):
        """Создаёт тензор из вложенных списков / ``Matrix`` / ``NDimArray`` / скаляра.

        Параметры
        ---------
        data
            Компоненты: ``int``/``float``/``sympy`` скаляр, вложенный список,
            ``sympy.Matrix``, ``NDimArray`` или другой ``Tensor``.
        name : str
            Имя тензора для отображения.
        symmetry : str | None
            Метка симметрии (см. :attr:`symmetry`).
        indices : Sequence[str] | None
            Имена индексов, по одному на ось.
        rank : int | None
            Если задан — требование к рангу (для ``Tensor.scalar``); иначе
            выводится из ``data``.

        Raises
        ------
        ValueError
            Несовпадение ``rank`` с реальной размерностью данных, либо длина ``indices``
            не равна рангу.
        """
        arr = _as_ndim_array(data)
        if rank is not None and arr.rank() != int(rank):
            raise ValueError(f"ожидался ранг {rank}, получены данные ранга {arr.rank()}")
        self._data = arr
        self._name = str(name)
        self._symmetry = None if symmetry is None else str(symmetry)
        if indices is None:
            self._indices: tuple[str, ...] | None = None
        else:
            self._indices = tuple(indices)
            if len(self._indices) != arr.rank():
                raise ValueError(
                    f"indices {self._indices} не соответствуют рангу {arr.rank()}"
                )

    # ------------------------------------------------------------- свойства

    @property
    def data(self) -> ImmutableDenseNDimArray:
        """Сырые компоненты (``ImmutableDenseNDimArray``)."""
        return self._data

    @property
    def rank(self) -> int:
        """Число индексов."""
        return self._data.rank()

    @property
    def shape(self) -> tuple[int, ...]:
        """Размерности по каждому индексу."""
        return tuple(self._data.shape)

    @property
    def symmetry(self) -> str | None:
        """Метка симметрии: ``'symmetric'`` / ``'antisymmetric'`` / ``'traceless'`` / ``'stf'`` / ``None``."""
        return self._symmetry

    @property
    def name(self) -> str:
        """Имя тензора."""
        return self._name

    @property
    def indices(self) -> tuple[str, ...] | None:
        """Имена индексов либо ``None``."""
        return self._indices

    @property
    def free_symbols(self) -> set:
        """Все свободные символы во всех компонентах."""
        return set().union(*(sp.sympify(c).free_symbols for c in self)) if len(self) else set()

    # ------------------------------------------------------- конструкторы

    @classmethod
    def scalar(cls, expr, name: str = "T") -> "Tensor":
        """Тензор ранга 0 из скалярного выражения."""
        return cls(sp.sympify(expr), name=name, rank=0)

    @classmethod
    def zeros(cls, shape: Sequence[int], name: str = "0") -> "Tensor":
        """Нулевой тензор заданной формы."""
        shape = tuple(int(s) for s in shape)
        if not shape:
            return cls.scalar(sp.Integer(0), name=name)
        return cls(MutableDenseNDimArray.zeros(*shape), name=name)

    @classmethod
    def from_matrix(cls, mat: sp.MatrixBase, name: str = "T",
                    symmetry: str | None = None) -> "Tensor":
        """Тензор из ``sympy.Matrix`` (ранг 1 для столбца/строки, ранг 2 иначе)."""
        if mat.rows == 1 or mat.cols == 1:
            return cls([sp.sympify(v) for v in mat], name=name, symmetry=symmetry)
        return cls([[sp.sympify(v) for v in row] for row in mat.tolist()],
                   name=name, symmetry=symmetry)

    @classmethod
    def delta(cls, dim: int = 3, name: str = "delta") -> "Tensor":
        """Кронекеров дельта :math:`\\delta_{ij}` — метрика декартова базиса."""
        return cls(sp.eye(int(dim)), name=name, symmetry="symmetric")

    @classmethod
    def levi_civita(cls, dim: int = 3, name: str = "epsilon") -> "Tensor":
        """Тензор Леви-Чивиты :math:`\\varepsilon_{i_1...i_dim}` (полностью антисимметричный)."""
        dim = int(dim)
        out = MutableDenseNDimArray.zeros(*((dim,) * dim))
        for perm in itertools.permutations(range(dim)):
            out[perm] = _permutation_sign(perm)
        return cls(out, name=name, symmetry="antisymmetric")

    # --------------------------------------------------------- доступ/вывод

    def __getitem__(self, index) -> sp.Expr:
        """Компонент по кортежу индексов: ``T[0, 1, 2]``; ``T[()]`` для скаляра."""
        if isinstance(index, int):
            index = (index,)
        return sp.sympify(self._data[tuple(index)])

    def __iter__(self) -> Iterator[sp.Expr]:
        """Итерация по компонентам (для ранга 0 — один компонент)."""
        return iter(_flatten(self._data))

    def __len__(self) -> int:
        """Число компонентов."""
        return len(_flatten(self._data))

    def components(self) -> list:
        """Компоненты как вложенный список (скаляр — само выражение)."""
        if self.rank == 0:
            return sp.sympify(self._data[()])
        return self._data.tolist()

    def as_scalar(self) -> sp.Expr:
        """Скаляр (только для ранга 0)."""
        if self.rank != 0:
            raise ValueError(f"as_scalar: ранг {self.rank} != 0")
        return sp.sympify(self._data[()])

    def as_vector(self) -> sp.Matrix:
        """Столбец ``Matrix`` (только для ранга 1)."""
        if self.rank != 1:
            raise ValueError(f"as_vector: ранг {self.rank} != 1")
        return sp.Matrix([sp.sympify(c) for c in self._data])

    def as_matrix(self) -> sp.Matrix:
        """Матрица ``Matrix`` (только для ранга 2; ранг 1 приводится к столбцу)."""
        if self.rank == 1:
            return self.as_vector()
        if self.rank != 2:
            raise ValueError(f"as_matrix: ранг {self.rank} != 2")
        return self._data.tomatrix()

    def permute(self, order: Sequence[int], name: str | None = None) -> "Tensor":
        """Новый тензор с переставленными осями (см. :func:`_permute_array`)."""
        out = _permute_array(self._data, order)
        return Tensor(out, name=name or self._name, symmetry=self._symmetry)

    def _with_axes(self, axes: Sequence[int] | None, note: str) -> tuple[int, ...]:
        """Проверяет и нормализует список осей; ``None`` означает «все оси»."""
        if axes is None:
            return tuple(range(self.rank))
        axes = tuple(int(a) for a in axes)
        for a in axes:
            if not 0 <= a < self.rank:
                raise ValueError(f"{note}: ось {a} вне диапазона [0, {self.rank})")
        if len(set(axes)) != len(axes):
            raise ValueError(f"{note}: повтор индексов в {axes}")
        return axes

    # ------------------------------------------------------------- арифметика

    def _binary_shape_check(self, other: "Tensor", op: str) -> None:
        if self.shape != other.shape:
            raise ValueError(f"{op}: несовместимые формы {self.shape} и {other.shape}")

    def __add__(self, other):
        """Сложение тензоров одинаковой формы либо прибавление скаляра ко всем компонентам."""
        if isinstance(other, Tensor):
            self._binary_shape_check(other, "сложение")
            return Tensor(self._data + other._data, name=self._name,
                          symmetry=self._symmetry if self._symmetry == other._symmetry else None)
        scalar = sp.sympify(other)
        if isinstance(scalar, ImmutableDenseNDimArray) and scalar.rank() != 0:
            raise ValueError("сложение: ожидается тензор той же формы или скаляр")
        return Tensor(self._data + scalar, name=self._name, symmetry=self._symmetry)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            self._binary_shape_check(other, "вычитание")
            return Tensor(self._data - other._data, name=self._name,
                          symmetry=self._symmetry if self._symmetry == other._symmetry else None)
        return Tensor(self._data - sp.sympify(other), name=self._name, symmetry=self._symmetry)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return Tensor(-self._data, name=self._name, symmetry=self._symmetry)

    def __mul__(self, other):
        other = sp.sympify(other)
        if isinstance(other, ImmutableDenseNDimArray) and other.rank() != 0:
            raise ValueError("умножение Tensor на нематричный многомерный массив: используйте outer()")
        return Tensor(self._data * other, name=self._name, symmetry=self._symmetry)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = sp.sympify(other)
        return Tensor(self._data / other, name=self._name, symmetry=self._symmetry)

    # ------------------------------------------------------ произведения

    def outer(self, other: "Tensor", name: str | None = None) -> "Tensor":
        """Внешнее (тензорное) произведение: оси ``self``, затем оси ``other``.

        Ранг результата — сумма рангов. Метка симметрии сбрасывается в ``None``:
        симметрия отдельных групп осей не выражается одной строкой.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, name="S")
        data = sp.tensorproduct(self._data, other._data)
        return Tensor(_wrap(data), name=name or f"{self._name}*{other._name}", symmetry=None)

    def inner(self, other: "Tensor", axis_self: int, axis_other: int,
              name: str | None = None) -> "Tensor":
        """Свёртка одной пары осей между разными тензорами.

        Оси результата: оставшиеся оси ``self`` (в исходном порядке), затем
        оставшиеся оси ``other`` (в исходном порядке).
        """
        prod = self.outer(other)
        a = int(axis_self)
        b = int(axis_other) + self.rank
        if not 0 <= a < self.rank:
            raise ValueError(f"axis_self={axis_self} вне диапазона [0, {self.rank})")
        if not 0 <= b < prod.rank:
            raise ValueError(f"axis_other={axis_other} вне диапазона")
        if prod.shape[a] != prod.shape[b]:
            raise ValueError(
                f"inner: несовместимые размерности {prod.shape[a]} и {prod.shape[b]}"
            )
        return prod.contract(a, b, name=name)

    def contract(self, i: int, j: int, name: str | None = None) -> "Tensor":
        """Свёртка (индексное суммирование) по осям ``i`` и ``j`` — ранг ``-2``."""
        i = int(i)
        j = int(j)
        if not 0 <= i < self.rank or not 0 <= j < self.rank:
            raise ValueError(f"contract: оси ({i}, {j}) вне диапазона [0, {self.rank})")
        if i == j:
            raise ValueError("contract: оси должны быть различны")
        if self.shape[i] != self.shape[j]:
            raise ValueError(f"contract: размерности осей {i} и {j} различаются ({self.shape})")
        data = _wrap(sp.tensorcontraction(self._data, [i, j]))
        return Tensor(data, name=name or f"{self._name}(свёртка {i},{j})", symmetry=None)

    def full_inner(self, other: "Tensor", name: str | None = None) -> "Tensor":
        """Полное сжатие по всем осям: :math:`A^{i_1...i_k} B_{i_1...i_k}`.

        Требует одинаковой формы; возвращает тензор ранга 0 (скаляр). Оси
        сопоставляются по позициям: ось ``k`` одного тензора с осью ``k`` другого.
        """
        if self.rank != other.rank:
            raise ValueError(f"full_inner: ранги {self.rank} и {other.rank} различаются")
        if self.shape != other.shape:
            raise ValueError(f"full_inner: формы {self.shape} и {other.shape} различаются")
        if self.rank == 0:
            return Tensor(self.as_scalar() * other.as_scalar(),
                          name=name or f"({self._name}|{other._name})")
        prod = self.outer(other)
        k = self.rank
        for step in range(k):
            prod = prod.contract(0, k - step)
        return Tensor(prod, name=name or f"({self._name}|{other._name})", symmetry=None)

    def trace(self, i: int = 0, j: int = 1, name: str | None = None) -> "Tensor":
        """След по осям ``i``, ``j`` (по умолчанию — первые две). Скаляр для ранга 2."""
        return self.contract(i, j, name=name or f"Tr {self._name}")

    def matmul(self, other: "Tensor", name: str | None = None) -> "Tensor":
        """Матричное произведение: (2,2)->ранг 2, (2,1)->ранг 1, (1,1)->скаляр."""
        if self.rank == 2 and other.rank in (1, 2):
            return self.inner(other, 1, 0, name=name or f"{self._name}{other._name}")
        if self.rank == 1 and other.rank == 1:
            return self.inner(other, 0, 0, name=name or f"({self._name}|{other._name})")
        raise ValueError(f"matmul: неподходящие ранги ({self.rank}, {other.rank})")

    # --------------------------------------------------------- симметрии

    def symmetrize(self, indices: Sequence[int] | None = None, name: str | None = None) -> "Tensor":
        """Симметричная часть по осям ``indices`` (по умолчанию — все оси)."""
        axes = self._with_axes(indices, "symmetrize")
        data = _symmetrize_axes(self._data, axes)
        symmetry = "symmetric" if len(axes) == self.rank else self._symmetry
        return Tensor(data, name=name or f"{self._name}_sym", symmetry=symmetry)

    def antisymmetrize(self, indices: Sequence[int] | None = None, name: str | None = None) -> "Tensor":
        """Антисимметричная часть по осям ``indices`` (по умолчанию — все оси)."""
        axes = self._with_axes(indices, "antisymmetrize")
        data = _antisymmetrize_axes(self._data, axes)
        symmetry = "antisymmetric" if len(axes) == self.rank else self._symmetry
        return Tensor(data, name=name or f"{self._name}_asym", symmetry=symmetry)

    def traceless(self, i: int = 0, j: int = 1, metric: "Tensor | None" = None,
                  name: str | None = None) -> "Tensor":
        """Снятие следа по паре осей ``(i, j)``: :math:`T - \\delta_{ij} \\mathrm{Tr}_{ij}T / n`.

        Результат имеет нулевой след именно по этой паре осей (для ранга 2 это
        обычная бесследовая часть). ``metric`` — метрический тензор ранга 2 с
        одинаковыми размерностями по осям (по умолчанию :math:`\\delta_{ij}`).
        """
        i = int(i)
        j = int(j)
        if self.shape[i] != self.shape[j]:
            raise ValueError(f"traceless: размерности осей {i} и {j} различаются ({self.shape})")
        n = self.shape[i]
        g = Tensor.delta(n, name="g") if metric is None else Tensor(metric, name="g")
        if g.rank != 2 or g.shape != (n, n):
            raise ValueError(f"traceless: метрика должна иметь форму ({n}, {n}), получено {g.shape}")
        tr = self.contract(i, j)
        emb = g.outer(tr)  # оси: (g0, g1, tr...)
        rank = self.rank
        remaining = [p for p in range(rank) if p not in (i, j)]
        order = [0] * rank
        order[i] = 0
        order[j] = 1
        for k, pos in enumerate(remaining):
            order[pos] = 2 + k
        correction = emb.permute(order)
        out = self - correction / n
        symmetry = "traceless" if rank == 2 else None
        return Tensor(out, name=name or f"{self._name}_tl", symmetry=symmetry)

    def stf(self, indices: Sequence[int] | None = None, name: str | None = None) -> "Tensor":
        """Проекция Стэйси — Тейлор: симметричная + бесследовая часть (ВСЕ следы).

        Для тензора ранга ``k`` (по осям ``indices``) применяется точная
        комбинаторная формула :math:`\\sum_s c_s\\,\\mathrm{sym}(\\delta^{\\otimes s}
        \\otimes \\mathrm{Tr}^{(s)}T)` с коэффициентами :func:`stf_coefficients`,
        выведенными из условия обращения в нуль всех следов.

        Возвращает
        ----------
        Tensor
            Симметричный тензор с нулевыми следами по любой паре осей,
            ``symmetry == 'stf'``.
        """
        axes = self._with_axes(indices, "stf")
        if len(axes) < 2:
            raise ValueError("stf: нужно не менее двух осей")
        if any(self.shape[a] != self.shape[axes[0]] for a in axes):
            raise ValueError(f"stf: оси {axes} имеют разные размерности {self.shape}")
        if len(axes) != self.rank:
            raise ValueError("stf: пока поддерживается полная симметризация всех осей")
        dim = self.shape[0]
        coeffs = stf_coefficients(self.rank, dim)
        delta = Tensor.delta(dim)

        # Переставляем нужные оси в начало (здесь axes — все оси, значит порядок сохраняется)
        work = self.symmetrize(axes)
        result = Tensor(work, name="STF") * coeffs[0]
        traced = work
        for s in range(1, len(coeffs)):
            traced = traced.trace(0, 1)  # (s)-кратный след для симметричного тензора
            emb = delta
            for _ in range(s - 1):
                emb = emb.outer(delta)
            term = emb.outer(traced)
            term = term.symmetrize(tuple(range(self.rank)))
            result = result + term * coeffs[s]
        return Tensor(result, name=name or f"{self._name}_STF", symmetry="stf")

    def is_symmetric(self, i: int = 0, j: int = 1, tol_zero: bool = True) -> bool:
        """Проверка симметричности по паре осей (символьно, через ``simplify``)."""
        diff = self - self.permute([j if k == i else (i if k == j else k) for k in range(self.rank)])
        return all((sp.simplify(c) == 0) for c in diff)

    def is_antisymmetric(self, i: int = 0, j: int = 1) -> bool:
        """Проверка антисимметричности по паре осей."""
        swapped = self.permute([j if k == i else (i if k == j else k) for k in range(self.rank)])
        return all((sp.simplify(c) == 0) for c in self + swapped)

    def is_traceless(self, i: int = 0, j: int = 1) -> bool:
        """Проверка обращения в нуль следа по паре осей."""
        tr = self.trace(i, j)
        return all((sp.simplify(c) == 0) for c in tr)

    # ------------------------------------------------------ индекс/метрика

    def raise_index(self, i: int, metric: "Tensor | None" = None, name: str | None = None) -> "Tensor":
        """Подъём индекса ``i`` метрикой :math:`g^{ij}` (по умолчанию :math:`\\delta^{ij}`)."""
        return self._metric_action(i, metric, name or f"{self._name}^({i})")

    def lower_index(self, i: int, metric: "Tensor | None" = None, name: str | None = None) -> "Tensor":
        """Опускание индекса ``i`` метрикой :math:`g_{ij}` (по умолчанию :math:`\\delta_{ij}`)."""
        return self._metric_action(i, metric, name or f"{self._name}_{i}")

    def _metric_action(self, i: int, metric: "Tensor | None", name: str) -> "Tensor":
        """Свёрка заданной оси с метрическим тензором; поднятая ось встаёт на то же место."""
        i = int(i)
        if not 0 <= i < self.rank:
            raise ValueError(f"ось {i} вне диапазона [0, {self.rank})")
        n = self.shape[i]
        g = Tensor.delta(n, name="g") if metric is None else Tensor(metric, name="g")
        if g.rank != 2 or g.shape != (n, n):
            raise ValueError(f"метрический тензор должен иметь форму ({n}, {n}), получено {g.shape}")
        # prod оси: (g0, g1, self...) ; свёртка g1 с self-i
        prod = g.outer(self)
        self_axis_in_prod = i + 2
        contracted = prod.contract(1, self_axis_in_prod)  # оси: (g0, self остальные)
        # нужно вернуть поднятую ось на позицию i
        remaining = [p for p in range(self.rank) if p != i]
        order = [0] * self.rank
        order[i] = 0
        for k, pos in enumerate(remaining):
            order[pos] = 1 + k
        return contracted.permute(order, name=name)

    # ------------------------------------------------- преобразования данных

    def subs(self, substitutions: dict, name: str | None = None) -> "Tensor":
        """Покомпонентная подстановка; метаданные сохраняются."""
        data = _wrap(self._data.subs(substitutions))
        return Tensor(data, name=name or self._name, symmetry=self._symmetry)

    def xreplace(self, rule: dict, name: str | None = None) -> "Tensor":
        """Быстрая структурная подстановка ``xreplace`` по компонентам."""
        data = _wrap(self._data.xreplace(rule))
        return Tensor(data, name=name or self._name, symmetry=self._symmetry)

    def diff(self, symbol, *symbols, name: str | None = None) -> "Tensor":
        """Частная производная по символу (ранг не меняется)."""
        data = _wrap(sp.diff(self._data, symbol, *symbols))
        return Tensor(data, name=name or f"d{self._name}/d{symbol}", symmetry=None)

    def grad(self, coords: Sequence, name: str | None = None) -> "Tensor":
        """Градиент скалярного поля: для тензора ранга 0 — тензор ранга 1.

        Параметры
        ----------
        coords : Sequence
            Последовательность символов (обобщённых координат); результат имеет
            ось той же длины, что ``coords``.
        """
        if self.rank != 0:
            raise ValueError(f"grad: ожидается скаляр, получен ранг {self.rank}")
        expr = self.as_scalar()
        comps = [sp.diff(expr, c) for c in coords]
        return Tensor(comps, name=name or f"grad {self._name}", symmetry=None)

    def simplify(self, name: str | None = None) -> "Tensor":
        """``sp.simplify`` покомпонентно."""
        data = _wrap(self._data.simplify())
        return Tensor(data, name=name or self._name, symmetry=self._symmetry)

    def expand(self, **kwargs) -> "Tensor":
        """``sp.expand`` покомпонентно."""
        data = _wrap(self._data.applyfunc(lambda e: sp.expand(e, **kwargs)))
        return Tensor(data, name=self._name, symmetry=self._symmetry)

    def trigsimp(self, name: str | None = None) -> "Tensor":
        """``sp.trigsimp`` покомпонентно."""
        data = _wrap(self._data.applyfunc(sp.trigsimp))
        return Tensor(data, name=name or self._name, symmetry=self._symmetry)

    def applyfunc(self, func, name: str | None = None, keep_symmetry: bool = True) -> "Tensor":
        """Произвольная функция по компонентам."""
        data = _wrap(self._data.applyfunc(func))
        return Tensor(data, name=name or self._name,
                      symmetry=self._symmetry if keep_symmetry else None)

    def conjugate(self, name: str | None = None) -> "Tensor":
        """Комплексное сопряжение покомпонентно."""
        data = _wrap(self._data.applyfunc(sp.conjugate))
        return Tensor(data, name=name or f"{self._name}*", symmetry=self._symmetry)

    def real_part(self, name: str | None = None) -> "Tensor":
        """Действительная часть (вещественный тензор того же ранга)."""
        data = _wrap(self._data.applyfunc(sp.re))
        return Tensor(data, name=name or f"Re {self._name}", symmetry=self._symmetry)

    def imag_part(self, name: str | None = None) -> "Tensor":
        """Мнимая часть (вещественный тензор того же ранга)."""
        data = _wrap(self._data.applyfunc(sp.im))
        return Tensor(data, name=name or f"Im {self._name}", symmetry=self._symmetry)

    # ------------------------------------------------------------- сверки

    def equals(self, other: "Tensor", method: str = "simplify") -> bool:
        """Тождественное равенство по всем компонентам (после упрощения)."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.shape != other.shape:
            return False
        diff = self._data - other._data
        for c in _flatten(_wrap(diff)):
            d = sp.simplify(c) if method == "simplify" else sp.trigsimp(c)
            if d != 0:
                return False
        return True

    def __eq__(self, other):  # pragma: no cover - структурное, не математическое равенство
        if not isinstance(other, Tensor):
            return NotImplemented
        return (self._data == other._data and self.shape == other.shape
                and self._symmetry == other._symmetry)

    __hash__ = None  # тензоры изменяемы по смыслу (через операции) — не хэшируем

    # --------------------------------------------------------------- печать

    def __repr__(self) -> str:
        sym = "" if self._symmetry is None else f", symmetry={self._symmetry!r}"
        idx = "" if self._indices is None else f", indices={self._indices}"
        return f"Tensor(name={self._name!r}, rank={self.rank}, shape={self.shape}{sym}{idx})"

    def __str__(self) -> str:
        return f"{self._name}[{self.rank}]{self.shape}"

    def show(self, **kwargs) -> str:
        """``ascii``-представление компонентов (удобно для отладки)."""
        return sp.pretty(self._data, **kwargs) if self.rank else sp.pretty(self.as_scalar())


def tensor_from(data, name: str = "T", symmetry: str | None = None) -> Tensor:
    """Фабрика :class:`Tensor` (принимает скаляры, списки, ``Matrix``, ``Tensor``)."""
    return Tensor(data, name=name, symmetry=symmetry)
