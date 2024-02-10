import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import tf2_ros
from geometry_msgs.msg import TransformStamped
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from FreehandDrawer import FreehandDrawer
import torch.nn.functional as F
from utils import ImitationLearningModel

from animation import TrajectoryAnimator


class SquareTraj:
    def __init__(self, center_x=0.4, center_y=0, center_z=0.5, side_length=0.2, points_per_side=100):
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


model = ImitationLearningModel().to(device)

#<<<<-----------Model--------------->>>>

# model.load_state_dict(torch.load('models/diffusion_model.pth'))
model.load_state_dict(torch.load('/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/models/model_main.pth'))


def follow_trajectory(desired_trajectory):
    global end_effector_positions, record_position
    for point in desired_trajectory:
        desired_position = torch.tensor(point, dtype=torch.float32).unsqueeze(0).to(device)
        joint_angles = model(desired_position)
        joint_angles_np = joint_angles.cpu().detach().numpy().squeeze()
        move_to_joint_angles(joint_angles_np)
        
        # Update and record the current end-effector position if flag is set
        if record_position:
            update_end_effector_position()
            end_effector_positions.append(current_end_effector_position.copy())
        
        rospy.sleep(0.01)

if __name__ == "__main__":
    record_position = False  # Ensure it's initially set to False
    initial_joint_angles = [0.0, -0.785398163, 0.0, -2.35619449, 0, 1.57079632679, 0.785398163397]
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)
    
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)

    rospy.loginfo("Now The Robot is in Initial Position!!!!")


    end_effector_positions = [] 

    rospy.loginfo("Now Draw the Trajectory :) ")
    update_end_effector_position()  

    # drawer = FreehandDrawer()
    constant_z_height = 0.5
    # trajectory = drawer.draw_trajectory()

    square_trajectory_generator = SquareTraj(center_x=0.4, center_y=0, center_z=0.5, side_length=0.1, points_per_side=500)
    desired_trajectory = square_trajectory_generator.generate_trajectory()

    record_position = True  # Start recording positions
    follow_trajectory(desired_trajectory)  # Follow the square trajectory
    desired_trajectory = [np.array([x, y, z]) for x, y ,z in desired_trajectory]

    record_position = True  # Start recording positions
    follow_trajectory(desired_trajectory)  # Follow the trajectory

    rospy.loginfo("<<<<-----------Trajectory Successful--------------->>>>")
    
    # Animation
    if len(end_effector_positions) > 0:  # Ensure there are positions to animate
        animator = TrajectoryAnimator(end_effector_positions, desired_trajectory)
        animator.animate()
    else:
        rospy.loginfo("No positions recorded for animation.")
