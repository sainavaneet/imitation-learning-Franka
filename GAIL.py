
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
        desired_position_np = np.array([point[0], point[1], 0.4])  # Assuming you're adding a z-coordinate
        desired_position_tensor = torch.from_numpy(desired_position_np).float().to(device)  # Ensure it's a float tensor and move to device

        # Reshape the tensor to match the input shape expected by the model
        desired_position_tensor = desired_position_tensor.reshape(1, -1)  # Reshape to [1, number_of_features]

        # Use the actor model to predict the next action (joint angles) based on the desired position
        joint_angles = model.actor(desired_position_tensor)
        joint_angles_np = joint_angles.cpu().detach().numpy().squeeze()

        move_to_joint_angles(joint_angles_np)
        update_end_effector_position()
        end_effector_positions.append(current_end_effector_position.copy())
        rospy.sleep(0.1)


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




    drawer = FreehandDrawer()
    constant_z_height = 0.5
    trajectory = drawer.draw_trajectory()
    update_end_effector_position()
    record_position = True
    follow_trajectory(trajectory)
    rospy.loginfo("<<<<-----------Trajectory Successful--------------->>>>")
    if len(end_effector_positions) > 0:
        animator = TrajectoryAnimator(end_effector_positions, trajectory)
        animator.animate()
    else:
        rospy.loginfo("No positions recorded for animation.")