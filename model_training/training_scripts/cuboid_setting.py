import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Cuboid:
    """A cuboid with dimensions measured in meters."""

    def __init__(self, Lx, Ly, Lz, m, position, orientation):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.m = m
        self.position = np.array(position, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
        self.local_vertices = np.array([[-Lx/2, -Ly/2, -Lz/2],
                                   [Lx/2, -Ly/2, -Lz/2],
                                   [Lx/2, Ly/2, -Lz/2],
                                   [-Lx/2, Ly/2, -Lz/2],
                                   [-Lx/2, -Ly/2, Lz/2],
                                   [Lx/2, -Ly/2, Lz/2],
                                   [Lx/2, Ly/2, Lz/2],
                                   [-Lx/2, Ly/2, Lz/2]], dtype=float)
    


# each parameter here is written as Cuboid.{parameter name}
if __name__ == "__main__":
    cuboid = Cuboid(Lx=0.20, Ly=0.10, Lz=0.10, m=1.0, position=np.array([0.0, 0.0, 0.05]), orientation=np.eye(3))
    print(f"Cuboid dimensions: {cuboid.Lx} m x {cuboid.Ly} m x {cuboid.Lz} m")
    print(f"Cuboid mass:{cuboid.m} kg")
    world_vertices = cuboid.local_vertices @ cuboid.orientation.T + cuboid.position

    face_indices = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    faces = [world_vertices[indices] for indices in face_indices]

    plt.rcParams.update({"font.family": "DejaVu Sans", "toolbar": "None"})
    background = "#111214"
    fig = plt.figure(figsize=(9, 7), facecolor=background)
    fig.canvas.manager.set_window_title("Cuboid | Motion study")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(background)
    surface = Poly3DCollection(
        faces,
        facecolors=["#526172", "#cbd5e1", "#8293a8", "#60758e", "#94a7bd", "#71869f"],
        edgecolors="#dce5ee",
        linewidths=0.7,
    )
    ax.add_collection3d(surface)
    for coordinate in np.linspace(-0.18, 0.18, 9):
        ax.plot([-0.18, 0.18], [coordinate, coordinate], [0, 0], color="#30343b", linewidth=0.5)
        ax.plot([coordinate, coordinate], [-0.18, 0.18], [0, 0], color="#30343b", linewidth=0.5)
    ax.set_xlim(-0.18, 0.18)
    ax.set_ylim(-0.18, 0.18)
    ax.set_zlim(0.0, 0.20)
    ax.set_box_aspect((0.36, 0.36, 0.20))
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()
    ax.text(0.19, 0, 0, "x", color="#8b95a3", fontsize=10)
    ax.text(0, 0.19, 0, "y", color="#8b95a3", fontsize=10)
    fig.text(0.09, 0.90, "Cuboid", color="#f5f5f7", fontsize=28, weight="medium")
    fig.text(0.09, 0.85, "20 × 10 × 10 cm   /   2 kg", color="#969ca6", fontsize=11)
    fig.text(0.09, 0.09, "01   /   TRANSLATION STUDY", color="#969ca6", fontsize=9)
    fig.text(0.91, 0.09, "Drag to orbit", color="#969ca6", fontsize=9, ha="right")
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.82)
    plt.show()
