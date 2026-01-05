import pybullet as p
import pybullet_data
import time

TIME_STEP = 1/240

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(TIME_STEP)

robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

ARM_JOINTS = list(range(7))
EE_LINK_INDEX = 8

neutral = [0,-0.5,0,-2.2,0,2.0,0.8]
for j,q in zip(ARM_JOINTS,neutral):
    p.resetJointState(robot,j,q)

JOINT_LIMITS = {}
for j in ARM_JOINTS:
    p.changeDynamics(robot,j,linearDamping=0.04,angularDamping=0.04)
    info = p.getJointInfo(robot,j)
    lower,upper = info[8], info[9]
    JOINT_LIMITS[j] = (lower,upper)

def get_current_joint_positions():
    return [p.getJointState(robot,j) [0] for j in ARM_JOINTS]

MAX_DELTA = 0.02 # rad per step


def apply_arm_positions(q_target):
    q_current = get_current_joint_positions()

    for idx, j in enumerate(ARM_JOINTS):
        q_desired = q_target[j]
        q_now = q_current[idx]

        # Rate limit
        delta = q_desired - q_now
        delta = max(-MAX_DELTA, min(MAX_DELTA, delta))
        q_cmd = q_now + delta

        # Clamp to joint limits
        lower, upper = JOINT_LIMITS[j]
        q_cmd = max(lower, min(upper, q_cmd))

        p.setJointMotorControl2(
    bodyUniqueId=robot,
    jointIndex=j,
    controlMode=p.POSITION_CONTROL,
    targetPosition=q_cmd,
    force=500,
    positionGain=0.15,
    velocityGain=1.0
)


targets = [
    ([0.45,  0.00, 0.35], p.getQuaternionFromEuler([0, 3.14159, 0])),
    ([0.55,  0.15, 0.35], p.getQuaternionFromEuler([0, 3.14159, 0])),
    ([0.55, -0.15, 0.35], p.getQuaternionFromEuler([0, 3.14159, 0])),
    ([0.45,  0.00, 0.45], p.getQuaternionFromEuler([0, 3.14159, 0])),
]

i = 0
while True:
    pos,orn=targets[i]
    q = p.calculateInverseKinematics(
        robot,
        EE_LINK_INDEX,
        pos,
        orn,
        maxNumIterations=120,
        residualThreshold=1e-4
    )

    apply_arm_positions(q)

    for _ in range(240):
        p.stepSimulation()
        time.sleep(TIME_STEP)

    i = (i+1) % len(targets)


