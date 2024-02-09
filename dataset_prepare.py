import numpy as np
from kineamatics.panda_kinematics import PandaWithHandKinematics

# Initialize Kinematics
kinematics = PandaWithHandKinematics()

# Generate a simple trajectory for the dataset
# For simplicity, let's generate a straight line in the workspace of the robot
def generate_trajectory(start_position, end_position, steps=100):
    trajectory = np.linspace(start_position, end_position, steps)
    return trajectory

# Use IK to find joint angles for each desired position
def generate_dataset(trajectory):
    dataset = []
    current_joint_state = np.array([0, -0.7, 0, -2.35619449, 0, 1.57079632679, 0.785398163397])  # Initial joint configuration
    
    for position in trajectory:
        desired_end_effector_position = position
        solution = kinematics.ik(current_joint_state, desired_end_effector_position, np.array([0.0, 1.0, 0.0, 0.0]))
        if solution is not None:
            dataset.append((current_joint_state, desired_end_effector_position, solution))
            current_joint_state = solution  # Update current state to the solution for continuity in the dataset
        else:
            print("IK solution not found for a point in the trajectory.")
    
    return dataset

# Example trajectory: a straight line along the x-axis
start_position = np.array([0.6, 0.2, 0.5])  # Starting at x=0.3, y=0, z=0.5
end_position = np.array([0.4, 0.2, 0.5])  # Ending at x=0.5, y=0, z=0.5
trajectory = generate_trajectory(start_position, end_position, 100)

# Generate dataset
dataset = generate_dataset(trajectory)

# Optionally, save your dataset to a file for later use
np.save("datasets/robot_imitation_learning_dataset.npy", dataset)
