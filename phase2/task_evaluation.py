# task_evaluation.py - Phase 2: Task definition and success evaluation
#
# Task: Pick cube from initial position and place it in target zone
# Success criteria:
#   1. Cube was lifted (height > threshold)
#   2. Cube final position is within target zone
#   3. Optional: Completed within time limit

import numpy as np


# Initial cube position (where it starts)
INITIAL_CUBE_POS = [0.55, 0.0, 0.05]

# Target zone definition (where cube should end up)
TARGET_ZONE = {
    "center": [0.50, 0.15, 0.05],  # Different from start position
    "size": [0.10, 0.10, 0.02],    # 10cm x 10cm x 2cm box
}

# Success thresholds
LIFT_HEIGHT_THRESHOLD = 0.15  # Cube must be lifted at least 15cm
TIME_LIMIT = 20.0  # Maximum time in seconds (optional, set to None to disable)



def point_in_box(point, box_center, box_size):
    """
    Check if a point is within a box.
    
    Args:
        point: [x, y, z] point to check
        box_center: [x, y, z] center of box
        box_size: [dx, dy, dz] half-sizes of box
    
    Returns:
        bool: True if point is inside box
    """
    for i in range(3):
        if abs(point[i] - box_center[i]) > box_size[i] / 2:
            return False
    return True


def evaluate_episode(state_log):
    """
    Evaluate if an episode was successful.
    
    Args:
        state_log: Dictionary containing episode data with keys:
            - 'cube_positions': List of [x, y, z] positions over time
            - 'timestamps': List of timestamps (optional, for time limit check)
    
    Returns:
        dict: {
            'success': bool,
            'lifted': bool,
            'in_target_zone': bool,
            'within_time_limit': bool,
            'max_height': float,
            'final_position': [x, y, z],
            'target_zone_center': [x, y, z],
            'details': str
        }
    """
    if not state_log or 'cube_positions' not in state_log:
        return {
            'success': False,
            'lifted': False,
            'in_target_zone': False,
            'within_time_limit': False,
            'max_height': 0.0,
            'final_position': None,
            'target_zone_center': TARGET_ZONE["center"],
            'details': 'Invalid state log'
        }
    
    cube_positions = state_log['cube_positions']
    if len(cube_positions) == 0:
        return {
            'success': False,
            'lifted': False,
            'in_target_zone': False,
            'within_time_limit': False,
            'max_height': 0.0,
            'final_position': None,
            'target_zone_center': TARGET_ZONE["center"],
            'details': 'Empty episode'
        }
    
    # Convert to numpy for easier computation
    positions = np.array(cube_positions)
    
    # Check 1: Was cube lifted?
    max_height = float(np.max(positions[:, 2]))  # Z is height
    initial_height = float(positions[0, 2])
    lifted = max_height > (initial_height + LIFT_HEIGHT_THRESHOLD)
    
    # Check 2: Is final position in target zone?
    final_position = positions[-1].tolist()
    in_target_zone = point_in_box(
        final_position,
        TARGET_ZONE["center"],
        TARGET_ZONE["size"]
    )
    
    # Check 3: Time limit (if timestamps provided)
    within_time_limit = True
    if TIME_LIMIT is not None and 'timestamps' in state_log:
        timestamps = state_log['timestamps']
        if len(timestamps) > 0:
            duration = timestamps[-1] - timestamps[0]
            within_time_limit = duration <= TIME_LIMIT
    
    # Overall success: all criteria met
    success = lifted and in_target_zone and within_time_limit
    
    # Generate details string
    details_parts = []
    details_parts.append(f"Max height: {max_height:.3f}m (threshold: {initial_height + LIFT_HEIGHT_THRESHOLD:.3f}m)")
    details_parts.append(f"Final position: [{final_position[0]:.3f}, {final_position[1]:.3f}, {final_position[2]:.3f}]")
    details_parts.append(f"Target zone: center {TARGET_ZONE['center']}, size {TARGET_ZONE['size']}")
    if TIME_LIMIT is not None and 'timestamps' in state_log:
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0
        details_parts.append(f"Duration: {duration:.2f}s (limit: {TIME_LIMIT}s)")
    
    details = " | ".join(details_parts)
    
    return {
        'success': success,
        'lifted': lifted,
        'in_target_zone': in_target_zone,
        'within_time_limit': within_time_limit,
        'max_height': max_height,
        'final_position': final_position,
        'target_zone_center': TARGET_ZONE["center"],
        'target_zone_size': TARGET_ZONE["size"],
        'details': details
    }


def get_task_info():
    """Get task configuration information."""
    return {
        'task_name': 'Pick and Place',
        'description': 'Pick cube from initial position and place in target zone',
        'initial_cube_pos': INITIAL_CUBE_POS,
        'target_zone': TARGET_ZONE,
        'lift_height_threshold': LIFT_HEIGHT_THRESHOLD,
        'time_limit': TIME_LIMIT,
    }


def visualize_target_zone(visual_shape_id=None):
    """
    Create a visual representation of the target zone in PyBullet.
    
    Args:
        visual_shape_id: Optional existing visual shape ID to update
    
    Returns:
        int: Visual shape ID (can be used to update/remove later)
    """
    import pybullet as p
    
    # Create a box shape for the target zone
    box_size = TARGET_ZONE["size"]
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[s/2 for s in box_size],
        rgbaColor=[0, 1, 0, 0.3]  # Semi-transparent green
    )
    
    # Create a multi-body for the visual (or update existing)
    if visual_shape_id is None:
        target_zone_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=TARGET_ZONE["center"]
        )
        return target_zone_id
    else:
        # Update existing
        p.resetBasePositionAndOrientation(
            visual_shape_id,
            TARGET_ZONE["center"],
            [0, 0, 0, 1]
        )
        return visual_shape_id

