import numpy as np
import json
from FreehandDrawer import FreehandDrawer, PandaWithHandKinematics  

if __name__ == "__main__":
    drawer = FreehandDrawer()
    trajectory = drawer.draw_trajectory()
    interpolated_trajectory = drawer.interpolate_trajectory(num_points=10) 
 
    dataset = []
    calculate_ik = PandaWithHandKinematics()

    for point in interpolated_trajectory:
        desired_position = np.array([point[0], point[1], 0.5])
        orientation_quat = np.array([1.0, 1.0, 0.0, 0.0])
        initial_joint_positions = np.array([0, 0, 0, -1.57, 0, 1.57, 0.785])
        joint_angles = calculate_ik.ik(initial_joint_positions, desired_position, orientation_quat)
        
        if joint_angles is not None:
            dataset_entry = {
                "current_state": None,
                "desired_end_effector_position": desired_position.tolist(),
                "joint_angles": joint_angles.tolist()
            }
            dataset.append(dataset_entry)

    # Save the dataset to a JSON file
    file_name = "datasets/newdataset1.json"
    with open(file_name, "w") as json_file:
        json.dump(dataset, json_file)
    print("Dataset saved to", file_name)

    # Extracting 'desired_end_effector_position' and 'joint_angles' for saving to text files
    exp_states = np.array([entry["desired_end_effector_position"] for entry in dataset])
    exp_actions = np.array([entry["joint_angles"] for entry in dataset])

    # Save to text files
    np.savetxt("save_data_state2.txt", exp_states)
    np.savetxt("save_data_action2.txt", exp_actions)

    print("State and action data saved to 'save_data_state2.txt' and 'save_data_action2.txt'.")
