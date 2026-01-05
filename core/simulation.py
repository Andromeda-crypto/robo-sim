# simulation.py - Simulating a robotic arm using PyBullet

import pybullet as p
import pybullet_data
import time

class RoboticArm:
    def __init__(self, urdf_path, time_step=1./240.):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(time_step)
        p.setGravity(0, 0, -9.81)

        self.arm = p.loadURDF(urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.arm)

        print(f"Loaded robot with {self.num_joints} joints")

    def set_joint_position(self, joint_indices, target_positions):
        if len(joint_indices) != len(target_positions):
            raise ValueError("joint_indices and target_positions must be same length")

        for joint_index, target in zip(joint_indices, target_positions):
            if joint_index >= self.num_joints:
                raise IndexError(f"Joint {joint_index} does not exist")

            p.setJointMotorControl2(
                bodyUniqueId=self.arm,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=500
            )

    def step_simulation(self, steps=240):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1./240.)

    def disconnect(self):
        p.disconnect()


def main():
    # Use a robot WITH joints
    urdf_path = "kuka_iiwa/model.urdf"

    robotic_arm = RoboticArm(urdf_path)

    joint_indices = [0, 1, 2]
    target_positions = [0.5, -0.5, 0.5]

    robotic_arm.set_joint_position(joint_indices, target_positions)
    robotic_arm.step_simulation(steps=400)
    robotic_arm.disconnect()


if __name__ == "__main__":
    main()
