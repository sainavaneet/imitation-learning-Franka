import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import tf2_ros
from geometry_msgs.msg import TransformStamped
from panda_kinematics import PandaWithPumpKinematics
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from FreehandDrawer import FreehandDrawer
import torch.nn.functional as F

from utils import ImitationLearningModel

rospy.init_node('robot_trajectory_follower', anonymous=True)
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)
joints_publishers = [rospy.Publisher(f'/franka/panda_joint{i+1}_controller/command', Float64, queue_size=10) for i in range(7)]
kinematics = PandaWithPumpKinematics()
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
model.load_state_dict(torch.load('/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/model.pth'))

def follow_trajectory(desired_trajectory):
    for point in desired_trajectory:
        desired_position = torch.tensor(point, dtype=torch.float32).unsqueeze(0).to(device)
        joint_angles = model(desired_position)
        joint_angles_np = joint_angles.cpu().detach().numpy().squeeze()
        move_to_joint_angles(joint_angles_np)
        rospy.sleep(0.03)  

if __name__ == "__main__":
    initial_joint_angles = [0, -0.7, 0, -2.35619449, 0, 1.57079632679, 0.785398163397]
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)
    rospy.loginfo("Now The Robot is in Initial Position!!!!")
    rospy.loginfo("Now Draw the Trajectory :) ")

    update_end_effector_position()
    drawer = FreehandDrawer()
    constant_z_height = 0.5
    drawn_trajectory_2d = drawer.draw_trajectory()
    desired_trajectory = [np.array([x, y, constant_z_height]) for x, y in drawn_trajectory_2d]
    follow_trajectory(desired_trajectory)
    rospy.loginfo("<<<<-----------Trajectory Succefull--------------->>>>")
