from solver import Solver, NodeConv, Direction, NodeConstT
from nptyping import NDArray, Float64, Shape
import matplotlib.pyplot as plt
import numpy as np
import os

##################################
# Nam: ----
# Shomare Daneshjooii: ----

# How to run the project:
# 1. Install python (minimum version 3.11)
# 2. run command `pip install -r requirements.txt` to install dependencies
# 3. run command `python3 main.py`

# plots will be generated in the '/plots' directory and data will be generated in '/data' directory
##################################


def main():
    EXPORT_CSV = True
    EXPORT_PLOTS = True

    width = 3
    height = 39e-2
    x_grid = 101
    y_grid = 14

    # DEFINING CONST VARIABLES ##########################
    # initial temperature
    t_init = c2k(-5)
    k_s1 = 2.23
    k_s2 = 2.32
    k_h = 0.5
    k_i = 0.043
    k_c = 2.32

    ro_s1 = 2260
    ro_s2 = 2260
    ro_h = 1000
    ro_i = 16
    ro_c = 2260

    c_s1 = 880
    c_s2 = 880
    c_h = 4200
    c_i = 840
    c_c = 880

    s = Solver(width, height, x_grid, y_grid, k_c, ro_c, c_c, t_init)

    # DEFINING NODES WITH CONVECTION ####################
    conv_h = 20.0
    t_f = c2k(28.0)
    for x in range(0, x_grid):
        s.nodes[0][x].conv = NodeConv(conv_h, t_f, Direction.Y)

    i_start_y = 0
    i_end_y = 9
    i_start_x = 6
    i_end_x = x_grid - 6

    for y in range(i_start_y, i_end_y):
        for x in range(i_start_x, i_end_x):
            # INSIDE I NODES ####################################
            # we first set all nodes as i, but later in the code
            # we will redefine nodes properly
            if (i_start_x < x < i_end_x) and (i_start_y < y < i_end_y):
                s.nodes[y][x].cond.k = k_i
                s.nodes[y][x].transient.ro = ro_i
                s.nodes[y][x].transient.c = c_i
            # EDGE I NODES ######################################
            if (x == i_start_x or x == i_end_x and y < i_end_y) or (
                y == i_end_y and i_start_x <= x <= i_end_x
            ):
                s.nodes[y][x].cond.k = (k_i + k_c) / 2
                s.nodes[y][x].transient.ro = (ro_i + ro_c) / 2
                s.nodes[y][x].transient.c = (c_i + c_c) / 2
    s1_start_y = 0
    s1_end_y = 2
    s1_start_x = 7
    s1_end_x = 94
    h_start_y = 2
    h_end_y = 5
    h_start_x = s1_start_x
    h_end_x = s1_end_x

    s2_start_y = h_end_y
    s2_end_y = 7

    for y in range(s1_start_y, s1_end_y):
        for x in range(s1_start_x, s1_end_x):
            # INSIDE S1 + S2 ##############################
            # we will fill all the points with s1 and later fill them with h values
            if (s1_start_x < x < s1_end_x) and (s1_start_y < y < s2_end_y):
                s.nodes[y][x].cond.k = k_s1
                s.nodes[y][x].transient.ro = ro_s1
                s.nodes[y][x].transient.c = c_s1

            # EDGES OF S1 + S2 ##############################
            if (
                x == s1_start_x
                or x == i_end_x
                and y < s2_end_y
                and not (s1_end_y < y < s2_start_y)
            ) or (y == i_end_y and i_start_x <= x <= i_end_x):
                s.nodes[y][x].cond.k = (k_s1 + k_i) / 2
                s.nodes[y][x].transient.ro = (ro_s1 + ro_i) / 2
                s.nodes[y][x].transient.c = (c_s1 + c_i) / 2

    # HORIZONTAL EDGES OF H ############################
    for x in range(h_start_x, h_end_x):
        s.nodes[h_start_y][x].cond.k = (k_h + k_s1) / 2
        s.nodes[h_start_y][x].transient.ro = (ro_h + ro_s1) / 2
        s.nodes[h_start_y][x].transient.c = (c_h + c_s1) / 2
        s.nodes[h_end_y][x].cond.k = (k_h + k_s2) / 2
        s.nodes[h_end_y][x].transient.ro = (ro_h + ro_s2) / 2
        s.nodes[h_end_y][x].transient.c = (c_h + c_s2) / 2

    # VERTICAL EDGES OF H ##############################
    for y in range(h_start_y, h_end_y):
        s.nodes[y][h_start_x].cond.k = (k_h + k_i) / 2
        s.nodes[y][h_start_x].transient.ro = (ro_h + ro_i) / 2
        s.nodes[y][h_start_x].transient.c = (c_h + c_i) / 2
        s.nodes[y][h_end_x].cond.k = (k_h + k_i) / 2
        s.nodes[y][h_end_x].transient.ro = (ro_h + ro_i) / 2
        s.nodes[y][h_end_x].transient.c = (c_h + c_i) / 2

    # EDGES OF THE H ###################################
    edge_points = [
        (h_start_y, h_start_x),
        (h_start_y, h_end_x),
        (h_end_y, h_start_x),
        (h_end_y, h_end_x),
    ]

    for point in edge_points:
        s.nodes[point[0]][point[1]].cond.k = (k_h + k_s1 + k_i) / 3
        s.nodes[point[0]][point[1]].transient.ro = (ro_h + ro_s1 + ro_i) / 3
        s.nodes[point[0]][point[1]].transient.c = (c_h + c_s1 + c_i) / 3

    # EDGES OF H #######################################
    for y in range(h_start_y, h_end_y):
        for x in range(h_start_x, h_end_x):
            if (h_start_x < x < h_end_x) and (h_start_y < y < h_end_y):
                s.nodes[y][x].cond.k = k_h
                s.nodes[y][x].transient.ro = ro_h
                s.nodes[y][x].transient.c = c_h

    # CONST T OF EARTH #################################
    const_t_earth = c2k(-5.0)
    for y in range(0, y_grid):
        s.nodes[y][0].const = NodeConstT(const_t_earth)
        s.nodes[y][-1].const = NodeConstT(const_t_earth)

    for x in range(0, x_grid):
        s.nodes[-1][x].const = NodeConstT(const_t_earth)

    # CONST T OF H #####################################
    const_t_h = c2k(50)
    y_const_t_h = 4
    for x in range(h_start_x, h_end_x):
        s.nodes[y_const_t_h][x].const = NodeConstT(const_t_h)
    s.calc_base_coefficients()

    # GET RESULTS AND EXPORTING RESULTS #################
    max_iters = 10000
    time_step = 15 * 60
    threshold = 0.01
    avg = 16.55
    _, axis = plt.subplots(2)
    plot_max = c2k(50.0)
    plot_min = c2k(-5.0)

    for i in range(1, max_iters + 1):
        s.next(time_step)
        curr_avg = s.t.mean() - 273.15
        curr_max = s.t.max() - 273.15
        curr_min = s.t.min() - 273.15
        temps = np.reshape(s.t.copy(), (x_grid, y_grid))
        max_i = temps.argmax()
        max_i_y = max_i % y_grid  # type: ignore
        max_i += -max_i_y + (y_grid - max_i_y)
        print(
            f"{str(i).zfill(3)}: {curr_min:.3f} {curr_avg:.3f} {curr_max:.3f}(#{str(max_i).zfill(3)})"
        )

        if EXPORT_CSV:
            os.makedirs("data", exist_ok=True)
            export_csv(temps, x_grid, y_grid, f"data/t{i * time_step / 60}.csv")
        if EXPORT_PLOTS:
            os.makedirs("plots", exist_ok=True)
            x = np.linspace(0, width, x_grid)
            y = np.linspace(0, height, y_grid)

            filled_levels = np.linspace(plot_min, plot_max, 100)
            axis[0].clear()
            axis[0].contourf(x, y, np.flip(temps.T, 0), levels=filled_levels)
            axis[0].set_title(f"Filled {time_step * i / 60}")

            stroke_levels = np.linspace(plot_min, plot_max, 30)
            axis[1].clear()
            axis[1].contour(x, y, np.flip(temps.T, 0), levels=stroke_levels)
            axis[1].set_title(f"Stroke {time_step * i / 60}")

            plt.savefig(f"plots/t{i * time_step / 60}.png")
        if i != 1 and (abs(curr_avg - avg) / avg) < threshold:
            break


def export_csv(
    data: NDArray[Shape["*, *"], Float64], x_grid: int, y_grid: int, output: str
):
    i = 1
    file = open(output, mode="w")
    for x in range(0, x_grid):
        for y in range(1, y_grid + 1):
            # y is reversed because our grid is top to bottom but question's is bottom to top
            degree = data[x][y_grid - y] - 273.15
            file.write(f"{i},{degree:.3f}\n")
            i += 1
    file.close()


def c2k(c: float) -> float:
    return c + 273.15


if __name__ == "__main__":
    main()
