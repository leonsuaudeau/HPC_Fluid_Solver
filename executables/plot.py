import matplotlib.pyplot as plt
import numpy as np
from pyrr.trig import aspect_ratio
from rich import color


def calculate_viscosity(filename):
    seq = np.fromfile(filename, dtype=np.float32)

    N = len(seq)
    sum_t = 0
    sum_y = 0
    sum_tt = 0
    sum_ty = 0

    for t in range(N):
        y = np.log(abs(seq[t]))
        sum_t = sum_t + t
        sum_y = sum_y + y
        sum_tt = sum_tt + t * t
        sum_ty = sum_ty + t * y


    m = (N * sum_ty - sum_t * sum_y) / (N * sum_tt - sum_t * sum_t)
    k = 2 * np.pi / 32 # TODO: can this be made to automatically update?
    return -m / (k * k)

def plot_shear_wave():
    L = 128
    Y = np.arange(0, 1, 1/L)
    v_x = np.fromfile("outputs/moving_lid/v_x.bin", dtype=np.float32).reshape((L, L))

    f = v_x[10]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Y, f, label="x")
    plt.show()

def plot_viscosity():
    measured_values = []
    analytical_values = []
    omegas = []

    for o in range(1, 21):
        omega = 1 / o * 2
        name = "outputs/measurements_" + f"{o:02d}" + ".bin"
        nu_measured = calculate_viscosity(name)
        nu_analytical = (1 / omega - 0.5)/3

        measured_values.append(nu_measured)
        analytical_values.append(nu_analytical)
        omegas.append(omega)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omegas, analytical_values, label="nu_analytical")
    ax.plot(omegas, measured_values, label="nu_measured")
    ax.set_xlabel("omega")
    ax.set_ylabel("viscosity")
    ax.legend()

    plt.show()
    #print(f"Measured viscosity: {nu_measured}\nAnalytical solution: {nu_analytical}")

def plot_magnitude():
    width, height = 128, 128
    v_x = np.fromfile("outputs/moving_lid/v_x.bin", dtype=np.float32).reshape((width, height))
    v_y = np.fromfile("outputs/moving_lid/v_y.bin", dtype=np.float32).reshape((width, height))

    fig, ax = plt.subplots()
    mag = np.sqrt(v_x**2 + v_y**2)
    im = ax.imshow(mag, origin='lower', cmap='viridis')
    plt.colorbar(im)
    plt.show()

def plot_stream():
    width, height = 128, 128
    v_x = np.fromfile("outputs/moving_lid/v_x.bin", dtype=np.float32).reshape((width, height))
    v_y = np.fromfile("outputs/moving_lid/v_y.bin", dtype=np.float32).reshape((width, height))
    mag = np.sqrt(v_x**2 + v_y**2)

    x, y = np.meshgrid(np.arange(0, 1, 1/height), np.arange(0, 1, 1/width))
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    stream = ax.streamplot(x, y, v_x, v_y, color=mag, density=1.8, linewidth=1, arrowsize=1)
    fig.colorbar(stream.lines)
    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")
    plt.show()


if __name__ == "__main__":
    plot_shear_wave()