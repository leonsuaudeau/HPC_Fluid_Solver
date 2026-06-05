import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":
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



