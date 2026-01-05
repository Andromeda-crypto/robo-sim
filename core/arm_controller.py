# arm_controller.py
import time
import pybullet as p
import pybullet_data


class ArmController:
    """
    Pose-level controller for Franka Panda in PyBullet.
    External code commands end-effector target pose; this class:
      - solves IK
      - clamps to joint limits
      - rate-limits joint commands
      - sends motor commands
      - steps simulation
    """

    def __init__(
        self,
        gui=True,
        time_step=1 / 240,
        max_delta=0.02,           
        motor_force=500,
        position_gain=0.15,
        velocity_gain=1.0,
    ):
        self.time_step = time_step
        self.max_delta = max_delta
        self.motor_force = motor_force
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain

        self.cid = p.connect(p.GUI if gui else p.DIRECT)

        # PyBullet data path + search path 
        data_path = pybullet_data.getDataPath()
        print("pybullet_data path:", data_path)
        p.setAdditionalSearchPath(data_path)

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        # Load robot (raise immediately if it fails)
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        if self.robot < 0:
            raise FileNotFoundError(
                "Failed to load URDF 'franka_panda/panda.urdf'. "
                "Check that pybullet_data is installed correctly and the path above contains franka_panda/."
            )
        # Scene
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], useFixedBase=True)
        
        # Initial cube position (Phase 2: task definition)
        self.initial_cube_pos = [0.55, 0.0, 0.05]
        self.cube_id = p.loadURDF("cube_small.urdf", basePosition=self.initial_cube_pos)


        nj = p.getNumJoints(self.robot)
        print("Loaded robot id:", self.robot, "num joints:", nj)
        if nj < 7:
            raise RuntimeError(f"Robot has {nj} joints; expected Panda with at least 7.")

        # Arm joints (7 revolute)
        self.arm_joints = list(range(7))
        # End-effector link index (panda_hand)
        self.ee_link = 8
        self.gripper_joints = [9, 10]
        self.gripper_limits = {}
        for j in self.gripper_joints:
            info = p.getJointInfo(self.robot, j)
            self.gripper_limits[j] = (info[8], info[9])  


        # Cache joint limits
        self.joint_limits = {}
        for j in self.arm_joints:
            info = p.getJointInfo(self.robot, j)
            lower, upper = info[8], info[9]
            self.joint_limits[j] = (lower, upper)

        # Add damping to reduce oscillation after disturbances
        for j in self.arm_joints:
            p.changeDynamics(
                self.robot,
                linkIndex=j,
                linearDamping=0.04,
                angularDamping=0.04,
            )

        # Start in a neutral pose to help IK behave
        neutral = [0, -0.5, 0, -2.2, 0, 2.0, 0.8]
        for j, q in zip(self.arm_joints, neutral):
            p.resetJointState(self.robot, j, q)

        # Default target = None until user sets it
        self.target_pos = None
        self.target_orn = None

    def get_end_effector_pose(self):
        state = p.getLinkState(self.robot, self.ee_link)
        pos = state[4]  # worldLinkFramePosition
        orn = state[5]  # worldLinkFrameOrientation
        return list(pos), list(orn)

    def set_target_pose(self, pos, orn):
        """Set desired end-effector target pose (world frame)."""
        self.target_pos = list(pos)
        self.target_orn = list(orn)

    def _get_current_joint_positions(self):
        return [p.getJointState(self.robot, j)[0] for j in self.arm_joints]

    def _clamp(self, j, q):
        lo, hi = self.joint_limits[j]
        if q < lo:
            return lo
        if q > hi:
            return hi
        return q

    def _rate_limit(self, q_now, q_desired):
        delta = q_desired - q_now
        if delta > self.max_delta:
            delta = self.max_delta
        elif delta < -self.max_delta:
            delta = -self.max_delta
        return q_now + delta

    def step(self, sleep=True):
        """
        Run one control step toward target pose (if target set) and step physics.
        """
        if not p.isConnected():
            raise RuntimeError("PyBullet physics server is not connected (GUI likely closed/crashed). ")
        
        if self.target_pos is not None and self.target_orn is not None:
            q_ik = p.calculateInverseKinematics(
                self.robot,
                self.ee_link,
                self.target_pos,
                self.target_orn,
                maxNumIterations = 120,
                residualThreshold = 1e-4
            )

            q_now = self._get_current_joint_positions()

            for idx,j in enumerate(self.arm_joints):
                q_cmd = self._rate_limit(q_now[idx],q_ik[j])
                q_cmd = self._clamp(j, q_cmd)

                p.setJointMotorControl2(
                    bodyUniqueId=self.robot,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition = q_cmd,
                    force = self.motor_force,
                    positionGain=self.position_gain,
                    velocityGain = self.velocity_gain
                )
        p.stepSimulation()
        if sleep:
            time.sleep(self.time_step)


    def disconnect(self):
        """Cleanly disconnect from PyBullet physics server."""
        try:
            if p.isConnected():
                p.disconnect(self.cid)
        except Exception:
            # If disconnect fails, try disconnecting all connections
            try:
                p.disconnect()
            except Exception:
                pass

    def set_gripper(self, opening):
        """
        opening: 0.0 (closed) to 0.04 (open) approx for Panda fingers
        """
        opening = max(0.0, min(0.04, float(opening)))
        for j in self.gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=opening,
                force=100,
                positionGain=0.3,
                velocityGain=1.0,
            )
    
    def get_cube_pose(self):
        """
        Get current cube position and orientation.
        
        Returns:
            pos: [x, y, z] position
            orn: [qx, qy, qz, qw] orientation quaternion
        """
        pos, orn = p.getBasePositionAndOrientation(self.cube_id)
        return list(pos), list(orn)
    
    def get_all_joint_positions(self):
        """
        Get all joint positions including arm and gripper.
        
        Returns:
            arm_positions: List of 7 arm joint positions
            gripper_positions: List of 2 gripper joint positions
        """
        arm_pos = [p.getJointState(self.robot, j)[0] for j in self.arm_joints]
        gripper_pos = [p.getJointState(self.robot, j)[0] for j in self.gripper_joints]
        return arm_pos, gripper_pos
    
    def reset_scene(self, initial_cube_pos=None):
        """
        Reset scene to initial state for replay.
        
        Args:
            initial_cube_pos: Optional custom initial cube position.
                            If None, uses the original initial position.
        """
        if initial_cube_pos is None:
            initial_cube_pos = self.initial_cube_pos
        
        # Reset arm to neutral pose
        neutral = [0, -0.5, 0, -2.2, 0, 2.0, 0.8]
        for j, q in zip(self.arm_joints, neutral):
            p.resetJointState(self.robot, j, q)
        
        # Reset gripper to open
        for j in self.gripper_joints:
            p.resetJointState(self.robot, j, 0.04)
        
        # Reset cube to initial position
        p.resetBasePositionAndOrientation(
            self.cube_id,
            initial_cube_pos,
            [0, 0, 0, 1]  # No rotation
        )
        
        # Reset velocities to zero
        p.resetBaseVelocity(self.cube_id, [0, 0, 0], [0, 0, 0])
        for j in self.arm_joints + self.gripper_joints:
            p.resetJointState(self.robot, j, p.getJointState(self.robot, j)[0], 0.0)

