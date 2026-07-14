import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from enum import Enum

def plot_single():
    v_x =  np.fromfile("outputs/u_x.bin", dtype=np.float32).reshape((width, height))
    v_y =  np.fromfile("outputs/u_y.bin", dtype=np.float32).reshape((width, height))

    plt.imshow(v_x)
    plt.colorbar()
    plt.show()

def plot_single_stream():
    x, y = np.meshgrid(np.arange(0, height), np.arange(0, width))
    plt.streamplot(
        x, y, frames_y[10], frames_x[10], density=1.2, linewidth=1, arrowsize=1,
    )

    plt.gca().invert_yaxis()
    plt.show()

def animate_stream():
    x, y = np.meshgrid(np.arange(0, height), np.arange(0, width))
    fig, ax = plt.subplots()

    def update_frame(frame_idx):
        ax.clear()
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        u = frames_x[frame_idx]
        v = frames_y[frame_idx]
        stream = ax.streamplot(x, y, u, v, density=1.2, linewidth=1, arrowsize=1,)
        return [stream.lines, stream.arrows]

    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames_x), interval=frame_count, blit=False)
    plt.show()
    return anim

def animate_arrows(skip, scale):
    X, Y = np.meshgrid(
        np.arange(0, width, skip),
        np.arange(0, height, skip), indexing="xy"
    )

    fig, ax = plt.subplots()

    q = ax.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        frames_x[0][::skip, ::skip],
        frames_y[0][::skip, ::skip],
        scale=scale,
    )

    def update_frame(frame_idx):
        vx = frames_x[frame_idx]
        vy = frames_y[frame_idx]

        q.set_UVC(
            vx[::skip, ::skip],
            vy[::skip, ::skip]
        )
        return [q]

    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames_x), interval=frame_count, blit=True)
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

    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames_x), interval=frame_count, blit=True)
    plt.show()
    return anim

class AnimationType(Enum):
    arrows = 0
    magnitude = 1
    stream = 2

if __name__ == "__main__":
    width, height = 128, 128
    frame_count = 50
    animation_type = AnimationType.magnitude
    files_x = sorted(glob.glob("outputs/u_x_*.bin"))
    files_y = sorted(glob.glob("outputs/u_y_*.bin"))
    frames_x = [np.fromfile(file, dtype=np.float32).reshape((height, width)) for file in files_x]
    frames_y = [np.fromfile(file, dtype=np.float32).reshape((height, width)) for file in files_y]

    if animation_type == AnimationType.arrows:
        anim = animate_arrows(1, 5)
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
    elif animation_type == AnimationType.stream:
        anim = animate_stream()
        anim.save(
            "lbm_velocity.mp4",
            writer="ffmpeg",
            fps=20,
            dpi=200
        )
