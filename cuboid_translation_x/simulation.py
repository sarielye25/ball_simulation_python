# This script is used to translate a cuboid along the x-axis in a 3D space. The cuboid is defined by its dimensions and position, and the translation is performed by predicting a specified force in x direction. The script utilizes a simple physics model to calculate the new position of the cuboid, then appy RL for traning the policy agent.

from cuboid_setting import Cuboid
from physics import phy


if __name__ == "__main__":
    cuboid = Cuboid(
        Lx=0.20,
        Ly=0.10,
        Lz=0.10,
        m=2.0,
        position=[0.0, 0.0, 0.05],
        orientation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    applied_force = 10.0
    push_duration = 1.0
    kinetic_friction = 0.2
    static_friction = 0.4
    initial_x = cuboid.position[0]

    displacement = phy(
        applied_force, cuboid.m, push_duration, kinetic_friction, static_friction
    )
    cuboid.position[0] += displacement

    print(f"Initial x: {initial_x:.6f} m")
    print(f"Displacement: {displacement:.6f} m")
    print(f"Predicted stopping x: {cuboid.position[0]:.6f} m")

