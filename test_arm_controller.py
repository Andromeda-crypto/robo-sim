from arm_controller import ArmController
import pybullet as p

ctrl = ArmController(gui=True)

targets = [
    ([0.45,  0.00, 0.35], p.getQuaternionFromEuler([0, 3.14159, 0])),
    ([0.55,  0.15, 0.35], p.getQuaternionFromEuler([0, 3.14159, 0])),
    ([0.55, -0.15, 0.35], p.getQuaternionFromEuler([0, 3.14159, 0])),
    ([0.45,  0.00, 0.45], p.getQuaternionFromEuler([0, 3.14159, 0])),
]

i = 0
while True:
    pos, orn = targets[i]
    ctrl.set_target_pose(pos, orn)

    for _ in range(240):
        ctrl.step()

    i = (i+1) % len(targets)

    