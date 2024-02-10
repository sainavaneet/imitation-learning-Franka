import numpy as np
import json
from kineamatics.panda_kinematics import PandaWithHandKinematics


class CircleTrajectoryGenerator:
    def __init__(self, center_x=0.4, center_y=0, center_z=0, radius=0.3, points=10000):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.radius = radius
        self.points = points

    def generate_trajectory(self):
        theta = np.linspace(0, 2 * np.pi, self.points)
        y = self.radius * np.cos(theta) + self.center_y
        z = self.radius * np.sin(theta) + self.center_z
        x = np.full_like(y, self.center_x)
        return [np.array([x[i], y[i], z[i]]) for i in range(len(x))]

if __name__ == "__main__":
    # Replace FreehandDrawer with CircleTrajectoryGenerator
    circle_generator = CircleTrajectoryGenerator(center_x=0.5, center_y=0, center_z=0, radius=0.1, points=100)
    trajectory = circle_generator.generate_trajectory()
    print("Generated Circle Trajectory Points:")
    
    dataset = []
    calculate_ik = PandaWithHandKinematics()

    for point in trajectory:
        desired_position = point
        orientation_quat = np.array([1.0, 1.0, 0.0, 0.0])
        initial_joint_positions = np.array([0, 0, 0, 0, 0, 0, 0])
        joint_angles = calculate_ik.ik(initial_joint_positions, desired_position, orientation_quat)
        
        if joint_angles is not None:
            dataset_entry = {
                "current_state": None,
                "desired_end_effector_position": desired_position.tolist(),
                "joint_angles": joint_angles.tolist()
            }
            dataset.append(dataset_entry)

    file_name = "datasets/circle_trajectory_dataset.json"
    with open(file_name, "w") as json_file:
        json.dump(dataset, json_file)

    print("Dataset saved to", file_name)
