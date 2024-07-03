from dataclasses import dataclass
import numpy as np
from enum import Enum
from nptyping import NDArray, Float64, Shape
from typing import List


class Direction(Enum):
    X = 1
    Y = 2
    A = 3


@dataclass
class NodeCond:
    k: float


@dataclass
class NodeTransient:
    ro: float
    c: float


@dataclass
class NodeConv:
    h: float
    t_f: float
    surface: Direction


@dataclass
class NodeConstT:
    t: float


class Node:
    cond: NodeCond
    transient: NodeTransient
    conv: NodeConv | None
    const: NodeConstT | None

    def __init__(
        self,
        cond: NodeCond,
        transient: NodeTransient,
        conv: NodeConv | None = None,
        const: NodeConstT | None = None,
    ):
        self.cond = cond
        self.transient = transient
        self.conv = conv
        self.const = const

    def __str__(self) -> str:
        text = "Node("
        if self.cond is not None:
            text += str(self.cond) + ","
        if self.conv is not None:
            text += str(self.conv) + ","
        if self.const is not None:
            text += str(self.const) + ","
        return text + ")"


class Solver:
    nodes: List[List[Node]]
    x_grid: int
    y_grid: int
    width: float
    height: float
    init_t: float

    a_base: NDArray[Shape["*, *"], Float64]
    a: NDArray[Shape["*, *"], Float64]
    t: NDArray[Shape["*"], Float64]
    f_base: NDArray[Shape["*"], Float64]
    b_base: NDArray[Shape["*"], Float64]
    b: NDArray[Shape["*"], Float64]

    def __init__(
        self,
        width: float,
        height: float,
        x_grid: int,
        y_grid: int,
        k: float,
        transient_p: float,
        transient_c: float,
        init_t: float,
    ):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.width = width
        self.height = height
        self.init_t = init_t
        self.nodes = []
        for y in range(0, y_grid):
            self.nodes.append([])
            for _ in range(0, x_grid):
                self.nodes[y].append(
                    Node(NodeCond(k), NodeTransient(transient_p, transient_c))
                )

    def calc_base_coefficients(self):
        mat_size = self.x_grid * self.y_grid

        self.a_base = np.zeros((mat_size, mat_size), dtype=np.float64)
        self.t = np.full(mat_size, self.init_t, dtype=np.float64)
        self.b_base = np.zeros(mat_size, dtype=np.float64)
        self.f_base = np.zeros(mat_size, dtype=np.float64)

        grid_item_len = self.width / (self.x_grid - 1)
        grid_item_area = grid_item_len * grid_item_len

        for x in range(self.x_grid):
            for y in range(self.y_grid):
                i = x * self.y_grid + y

                node = self.nodes[y][x]

                if node.const is not None:
                    self.a_base[i][i] = 1
                    self.b_base[i] = node.const.t
                    self.t[i] = node.const.t
                    continue

                transient_area = grid_item_area
                if self.__is_corner(y, x):
                    transient_area /= 4
                elif self.__is_edge(y, x):
                    transient_area /= 2

                if self.__is_edge(y, x):
                    if self.__is_top_left_corner(y, x):
                        self.__set_cond_factors(i, 0.0, 0.5, 0.5, 0.0)
                        self.__set_transient_factors(
                            i, transient_area, 0.0, 0.5, 0.5, 0.0
                        )
                    elif self.__is_top_right_corner(y, x):
                        self.__set_cond_factors(i, 0.0, 0.0, 0.5, 0.5)
                        self.__set_transient_factors(
                            i, transient_area, 0.0, 0.0, 0.5, 0.5
                        )
                    elif self.__is_bottom_left_corner(y, x):
                        self.__set_cond_factors(i, 0.5, 0.5, 0.0, 0.0)
                        self.__set_transient_factors(
                            i, transient_area, 0.5, 0.5, 0.0, 0.0
                        )
                    elif self.__is_bottom_right_corner(y, x):
                        self.__set_cond_factors(i, 0.5, 0.0, 0.0, 0.5)
                        self.__set_transient_factors(
                            i, transient_area, 0.5, 0.0, 0.0, 0.5
                        )
                    elif y == 0:
                        self.__set_cond_factors(i, 0.0, 0.5, 1.0, 0.5)
                        self.__set_transient_factors(
                            i, transient_area, 0.0, 0.5, 1.0, 0.5
                        )
                    elif y == self.y_grid - 1:
                        self.__set_cond_factors(i, 1.0, 0.5, 0.0, 0.5)
                        self.__set_transient_factors(
                            i, transient_area, 1.0, 0.5, 0.0, 0.5
                        )
                    elif x == 0:
                        self.__set_cond_factors(i, 0.5, 1.0, 0.5, 0.0)
                        self.__set_transient_factors(
                            i, transient_area, 0.5, 1.0, 0.5, 0.0
                        )
                    elif x == self.x_grid - 1:
                        self.__set_cond_factors(i, 0.5, 0.0, 0.5, 1.0)
                        self.__set_transient_factors(
                            i, transient_area, 0.5, 0.0, 0.5, 1.0
                        )
                else:
                    self.__set_cond_factors(i, 1.0, 1.0, 1.0, 1.0)
                    self.__set_transient_factors(i, transient_area, 1.0, 1.0, 1.0, 1.0)

                if node.conv is not None:
                    area = self.__calc_conv_area(
                        i, node.conv.surface, grid_item_len, grid_item_len
                    )
                    self.a_base[i][i] -= node.conv.h * area
                    self.b_base[i] -= node.conv.h * area * node.conv.t_f

    def next(self, time_delta: float):
        self.a = self.a_base.copy()
        self.b = self.b_base.copy()
        for x in range(self.x_grid):
            for y in range(self.y_grid):
                i = x * self.y_grid + y

                fac = self.f_base[i] / time_delta
                self.a[i][i] -= fac
                self.b[i] -= fac * self.t[i]

        self.t = np.linalg.solve(self.a, self.b)

    def __set_transient_factors(
        self,
        i: int,
        area: float,
        top_f: float,
        right_f: float,
        bottom_f: float,
        left_f: float,
    ):
        x = i // self.y_grid
        y = i % self.y_grid

        ro_mean = 0.0
        c_mean = 0.0
        if top_f != 0.0:
            ro_mean += top_f * self.nodes[y - 1][x].transient.ro
            c_mean += top_f * self.nodes[y - 1][x].transient.c
        if bottom_f != 0.0:
            ro_mean += bottom_f * self.nodes[y + 1][x].transient.ro
            c_mean += bottom_f * self.nodes[y + 1][x].transient.c
        if right_f != 0.0:
            ro_mean += right_f * self.nodes[y][x + 1].transient.ro
            c_mean += right_f * self.nodes[y][x + 1].transient.c
        if left_f != 0.0:
            ro_mean += left_f * self.nodes[y][x - 1].transient.ro
            c_mean += left_f * self.nodes[y][x - 1].transient.c

        ro_mean /= top_f + right_f + left_f + bottom_f
        c_mean /= top_f + right_f + left_f + bottom_f

        self.f_base[i] = c_mean * ro_mean * area

    def __set_cond_factors(
        self,
        i: int,
        top_f: float,
        right_f: float,
        bottom_f: float,
        left_f: float,
    ):
        x = i // self.y_grid
        y = i % self.y_grid

        top = top_f * self.nodes[y - 1][x].cond.k
        right = right_f * self.nodes[y][x + 1].cond.k
        bottom = bottom_f * self.nodes[y + 1][x].cond.k
        left = left_f * self.nodes[y][x - 1].cond.k

        self.a_base[i][i] = -1 * (top + right + bottom + left)
        if top_f != 0.0:
            self.a_base[i][i - 1] = top
        if right_f != 0.0:
            self.a_base[i][i + self.y_grid] = right
        if bottom_f != 0.0:
            self.a_base[i][i + 1] = bottom
        if left_f != 0.0:
            self.a_base[i][i - self.y_grid] = left

    def __calc_conv_area(
        self, i: int, surface: Direction, x_size: float, y_size: float
    ) -> float:
        x = i // self.y_grid
        y = i % self.y_grid
        if surface == Direction.A:
            a = x_size * y_size
            if self.__is_corner(y, x):
                return a / 4.0
            return a / 2.0 if self.__is_edge(y, x) else a
        elif surface == Direction.X:
            return x_size / 2.0 if self.__is_corner(y, x) else x_size
        elif surface == Direction.Y:
            return y_size / 2.0 if self.__is_corner(y, x) else y_size
        assert False, "unknown convection surface type"

    def __is_edge(self, y, x) -> bool:
        return x == 0 or x == self.x_grid - 1 or y == 0 or y == self.y_grid - 1

    def __is_corner(self, y, x):
        return (
            self.__is_top_right_corner(y, x)
            or self.__is_top_left_corner(y, x)
            or self.__is_bottom_left_corner(y, x)
            or self.__is_bottom_right_corner(y, x)
        )

    @staticmethod
    def __is_top_left_corner(y, x) -> bool:
        return y == 0 and x == 0

    def __is_top_right_corner(self, y, x) -> bool:
        return y == 0 and x == self.x_grid - 1

    def __is_bottom_left_corner(self, y, x) -> bool:
        return x == 0 and y == self.y_grid - 1

    def __is_bottom_right_corner(self, y, x) -> bool:
        return y == self.y_grid - 1 and x == self.x_grid - 1
