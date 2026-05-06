import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from enum import Enum

def plot_single():
    v_x =  np.fromfile("outputs/v_x.bin", dtype=np.float32).reshape((width, height))
    v_y =  np.fromfile("outputs/v_y.bin", dtype=np.float32).reshape((width, height))

    plt.imshow(v_x)
    plt.colorbar()
    plt.show()

def animate_arrows(skip, scale):
    X, Y = np.meshgrid(
        np.arange(0, height, skip),
        np.arange(0, width, skip)
    )

    fig, ax = plt.subplots()

    q = ax.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        frames_y[0][::skip, ::skip],
        frames_x[0][::skip, ::skip],
        scale=scale,
    )

    def update_frame(frame_idx):
        # flipping to make the arrows point in the right direction
        vx = frames_y[frame_idx]
        vy = frames_x[frame_idx]

        q.set_UVC(
            vx[::skip, ::skip],
            vy[::skip, ::skip]
        )
        return [q]

    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames_x), interval=100, blit=True)
    plt.show()
    return anim

def animate_magnitude():
    fig, ax = plt.subplots()
    mag_0 = np.sqrt(frames_x[0]**2 + frames_y[0]**2)
    im = ax.imshow(mag_0, origin='lower', cmap='viridis')
    plt.colorbar(im)

    def update_frame(frame_idx):
        mag = np.sqrt(frames_x[frame_idx]**2 + frames_y[frame_idx]**2)
        im.set_array(mag)
        return [im]

    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames_x), interval=100, blit=True)
    plt.show()
    return anim

class AnimationType(Enum):
    arrows = 0
    magnitude = 1

if __name__ == "__main__":
    width, height = 32, 32
    animation_type = AnimationType.magnitude
    files_x = sorted(glob.glob("outputs/v_x_*.bin"))
    files_y = sorted(glob.glob("outputs/v_y_*.bin"))
    frames_x = [np.fromfile(file, dtype=np.float32).reshape((width, height)) for file in files_x]
    frames_y = [np.fromfile(file, dtype=np.float32).reshape((width, height)) for file in files_y]

    if animation_type == AnimationType.arrows:
        anim = animate_arrows(1, 50)
        anim.save(
            "lbm_velocity.mp4",
            writer="ffmpeg",
            fps=30,
            dpi=200
        )
    elif animation_type == AnimationType.magnitude:
        anim = animate_magnitude()
        anim.save(
            "lbm_velocity.mp4",
            writer="ffmpeg",
            fps=20,
            dpi=200
        )