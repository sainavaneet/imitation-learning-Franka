
from animation import TrajectoryAnimator
import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import tf2_ros
from geometry_msgs.msg import TransformStamped
import torch
from PGAIL import GAIL
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from Experttraj import ExpertTraj 
from FreehandDrawer import FreehandDrawer
import matplotlib.pyplot as plt
import numpy as np


class SquareTraj:
    def __init__(self, center_x, center_y, center_z, side_length, points_per_side):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.side_length = side_length
        self.points_per_side = points_per_side

    def generate_trajectory(self):
        half_side = self.side_length / 2
        corners = [
            (self.center_x - half_side, self.center_y - half_side),
            (self.center_x + half_side, self.center_y - half_side),
            (self.center_x + half_side, self.center_y + half_side),
            (self.center_x - half_side, self.center_y + half_side),
        ]
        
        # Generate points along the square's perimeter
        trajectory = []
        for i in range(len(corners)):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % len(corners)]
            xs = np.linspace(start_corner[0], end_corner[0], self.points_per_side, endpoint=False)
            ys = np.linspace(start_corner[1], end_corner[1], self.points_per_side, endpoint=False)
            zs = np.full_like(xs, self.center_z)
            for x, y, z in zip(xs, ys, zs):
                trajectory.append(np.array([x, y, z]))
        
        return trajectory
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

end_effector_positions = []
record_position = False
rospy.init_node('robot_trajectory_follower', anonymous=True)
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)
joints_publishers = [rospy.Publisher(f'/franka/panda_joint{i+1}_controller/command', Float64, queue_size=10) for i in range(7)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_robot_state = np.zeros(7)
current_end_effector_position = np.zeros(3)

def update_robot_state(msg):
    global current_robot_state
    current_robot_state = np.array(msg.position)

rospy.Subscriber('/franka/joint_states', JointState, update_robot_state)

def update_end_effector_position():
    global current_end_effector_position
    try:
        trans = tf_buffer.lookup_transform('panda_link0', 'panda_hand', rospy.Time(0))
        current_end_effector_position = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("TF lookup failed")

def move_to_joint_angles(joint_angles):
    for i, angle in enumerate(joint_angles):
        joints_publishers[i].publish(Float64(data=angle))
    rospy.sleep(0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env_name = "Franka"
state_dim = 3
action_dim = 7
max_action = 2
lr = 0.0002
betas = (0.5, 0.999)
batch_size = 100

# Initialize GAIL
model = GAIL(env_name, state_dim, action_dim, max_action, lr, betas, batch_size)

# Load models
epoch_to_load = 3000  # Example epoch number of the model you want to load
actor_path = f"./preTrained/GAIL_pretrained_{epoch_to_load}_actor.pth"
discriminator_path = f"./preTrained/GAIL_pretrained_{epoch_to_load}_discriminator.pth"

# Ensure the model is on the correct device


# Load the actor model
try:
    model.actor.load_state_dict(torch.load(actor_path, map_location=device))
    print("Actor model loaded successfully.")
except FileNotFoundError:
    print(f"Actor model file not found: {actor_path}")

# Load the discriminator model
try:
    model.discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    print("Discriminator model loaded successfully.")
except FileNotFoundError:
    print(f"Discriminator model file not found: {discriminator_path}")

def follow_trajectory(desired_trajectory):
    global end_effector_positions
    for point in desired_trajectory:
        # Convert the NumPy array to a PyTorch tensor
        desired_position_np = np.array([point[0], point[1], 0.5])  # Assuming you're adding a z-coordinate
        desired_position_tensor = torch.from_numpy(desired_position_np).float().to(device)  # Ensure it's a float tensor and move to device

        # Reshape the tensor to match the input shape expected by the model
        desired_position_tensor = desired_position_tensor.reshape(1, -1)  # Reshape to [1, number_of_features]

        # Use the actor model to predict the next action (joint angles) based on the desired position
        joint_angles = model.actor(desired_position_tensor)
        joint_angles_np = joint_angles.cpu().detach().numpy().squeeze()

        move_to_joint_angles(joint_angles_np)
        update_end_effector_position()
        end_effector_positions.append(current_end_effector_position.copy())
        rospy.sleep(0.01)


if __name__ == "__main__":
    record_position = False
    initial_joint_angles = [0.0, -0.785398163, 0.0, -2.35619449, 0, 1.57079632679, 0.785398163397]
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)
    rospy.loginfo("Now The Robot is in Initial Position!!!!")
    end_effector_positions = []
    rospy.loginfo("Now Draw the Trajectory :) ")


    square_trajectory_generator = SquareTraj(center_x=0.4, center_y=0, center_z=0.5, side_length=0.1, points_per_side=200)
    desired_trajectory = square_trajectory_generator.generate_trajectory()

    # drawer = FreehandDrawer()
    constant_z_height = 0.5
    # trajectory = drawer.draw_trajectory()
    update_end_effector_position()
    record_position = True
    follow_trajectory(desired_trajectory)
    rospy.loginfo("<<<<-----------Trajectory Successful--------------->>>>")
    end_effector_positions_np = np.array(end_effector_positions)
    desired_trajectory_np = np.array(desired_trajectory)

    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # X positions
    ax[0].plot(desired_trajectory_np[:, 0], label='Desired X')
    ax[0].plot(end_effector_positions_np[:, 0], label='Actual X')
    ax[0].set_title('X Coordinate')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Position')
    ax[0].legend()

    # Y positions
    ax[1].plot(desired_trajectory_np[:, 1], label='Desired Y')
    ax[1].plot(end_effector_positions_np[:, 1], label='Actual Y')
    ax[1].set_title('Y Coordinate')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Position')
    ax[1].legend()

    # Z positions
    ax[2].plot(desired_trajectory_np[:, 2], label='Desired Z')
    ax[2].plot(end_effector_positions_np[:, 2], label='Actual Z')
    ax[2].set_title('Z Coordinate')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Position')
    ax[2].legend()

    plt.tight_layout()
    plt.show()
