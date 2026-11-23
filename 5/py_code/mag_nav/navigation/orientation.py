"""Ориентация: пассивные вращения реализованы, *оценка* углов — заглушка.

Согласно ТЗ текущей итерации углы ориентации приёмника **не определяются**.
Поэтому:

* :class:`RotationMatrix` — полноценная чистая утилита пассивного поворота
  (нужна для согласования рамок «приёмник ↔ маяк»);
* :func:`solve_orientation` — явный ``NotImplementedError``.

Соглашение по углам (для будущей реализации)
--------------------------------------------
Внутренние (intrinsic) повороты ZYX: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
Это соответствует ``CalcRotMatrix.m`` / ``Ang2Mat.m``.
Обратное извлечение — ``CalcAngles.m`` (:meth:`RotationMatrix.to_euler`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import as_points, unit

__all__ = ["RotationMatrix", "solve_orientation", "OrientationEstimate"]


@dataclass(frozen=True)
class RotationMatrix:
    """Матрица поворота ``(3, 3)`` (пассивное преобразование координат)."""

    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError(f"Ожидалась матрица (3, 3), получено {m.shape}")
        object.__setattr__(self, "matrix", m)

    # -- конструкторы -------------------------------------------------------- #
    @classmethod
    def identity(cls) -> "RotationMatrix":
        return cls(np.eye(3))

    @classmethod
    def from_matrix(cls, m: ArrayLike, *, validate: bool = True) -> "RotationMatrix":
        r = cls(np.asarray(m, dtype=np.float64).reshape(3, 3))
        if validate:
            r.validate()
        return r

    @classmethod
    def from_euler(
        cls, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, *, degrees: bool = False
    ) -> "RotationMatrix":
        """``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
        if degrees:
            roll, pitch, yaw = np.radians([roll, pitch, yaw])
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
        return cls(rz @ ry @ rx)

    @classmethod
    def from_axis_angle(cls, axis: ArrayLike, angle: float) -> "RotationMatrix":
        """Формула Родригеса."""
        k = unit(np.asarray(axis, dtype=np.float64))
        c, s = np.cos(angle), np.sin(angle)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
        return cls(np.eye(3) + s * K + (1.0 - c) * (K @ K))

    @classmethod
    def align_vectors(cls, src: ArrayLike, dst: ArrayLike) -> "RotationMatrix":
        """Минимальный поворот, переводящий ``src`` в ``dst``."""
        a = unit(np.asarray(src, dtype=np.float64))
        b = unit(np.asarray(dst, dtype=np.float64))
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        if np.isclose(c, -1.0):
            ortho = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
            ax = unit(np.cross(a, ortho))
            return cls.from_axis_angle(ax, np.pi)
        s = float(np.linalg.norm(v))
        if s < 1e-14:
            return cls.identity()
        return cls.from_axis_angle(v / s, np.arctan2(s, c))

    # -- операции ------------------------------------------------------------ #
    def apply(self, vec: ArrayLike) -> NDArray[np.float64]:
        """``R @ v`` для ``(..., 3)``."""
        return as_points(vec) @ self.matrix.T

    def apply_inverse(self, vec: ArrayLike) -> NDArray[np.float64]:
        """``R.T @ v`` (для ортонормированной ``R`` это обратный поворот)."""
        return as_points(vec) @ self.matrix

    def inverse(self) -> "RotationMatrix":
        return RotationMatrix(self.matrix.T)

    def __matmul__(self, other: "RotationMatrix") -> "RotationMatrix":
        if not isinstance(other, RotationMatrix):
            return NotImplemented
        return RotationMatrix(self.matrix @ other.matrix)

    @property
    def T(self) -> "RotationMatrix":
        return self.inverse()

    def to_euler(self) -> tuple[float, float, float]:
        """``(roll, pitch, yaw)`` в радианах для конвенции ZYX."""
        m = self.matrix
        # R = Rz(yaw) Ry(pitch) Rx(roll): m[2,0] = -sin(pitch)
        pitch = np.arcsin(np.clip(-m[2, 0], -1.0, 1.0))
        if np.isclose(abs(np.cos(pitch)), 0.0):   # шарнирный замок
            roll = np.arctan2(-m[1, 2], m[1, 1])
            yaw = 0.0
        else:
            roll = np.arctan2(m[2, 1], m[2, 2])
            yaw = np.arctan2(m[1, 0], m[0, 0])
        return float(roll), float(pitch), float(yaw)

    def validate(self, tol: float = 1e-9) -> None:
        """Проверить ортонормированность и ``det = +1``."""
        m = self.matrix
        if not np.allclose(m @ m.T, np.eye(3), atol=tol):
            raise ValueError("Матрица не ортонормирована")
        if abs(np.linalg.det(m) - 1.0) > tol:
            raise ValueError("Определитель не равен +1 (это отражение?)")

    def deviation_from_identity(self) -> float:
        """Угол поворота относительно единичной матрицы, рад."""
        return float(np.arccos(np.clip((np.trace(self.matrix) - 1.0) / 2.0, -1.0, 1.0)))


IDENTITY: Final[RotationMatrix] = RotationMatrix.identity()


# --------------------------------------------------------------------------- #
# Оценка ориентации — НЕ реализована в текущей итерации
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrientationEstimate:
    """Заполнитель результата оценки ориентации (для будущей сигнатуры)."""

    rotation: RotationMatrix
    roll: float
    pitch: float
    yaw: float
    residual: float


def solve_orientation(
    a_vec: ArrayLike,
    b_vec: ArrayLike,
    position: ArrayLike,
    *,
    initial: RotationMatrix | None = None,
) -> OrientationEstimate:
    """Определение углов ориентации приёмника по a/b-параметрам.

    .. warning::
       **Не реализовано** в текущей версии ТЗ: углы не определяются,
       предполагается, что рамка приёмника совпадает с рамкой маяка
       (``RotationMatrix.identity()``).

    План будущей реализации (по ``CalcRotMatrix.m``, ``CalcAngles.m``,
    ``AnglesFromAngles.m``):

    1. По измеренным ``a_meas, b_meas`` (в рамке приёмника) и известной
       позиции ``(u, v, w)`` построить целевые ``a_ref = 3u*r_hat - x_hat``,
       ``b_ref = 3v*r_hat - y_hat``.
    2. Решить задачу ортогонального Прокрустеса: найти ``R``, минимизирующее
       ``||R @ [a_meas b_meas] - [a_ref b_ref]||_F`` через SVD.
    3. Извлечь углы :meth:`RotationMatrix.to_euler`, нормализовать
       (``AngNormal.m``).
    """
    raise NotImplementedError(
        "Определение ориентации отключено в текущей итерации ТЗ. "
        "Используйте RotationMatrix.identity() для согласования рамок."
    )