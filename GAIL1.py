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

from animation import TrajectoryAnimator


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Improved Generator (Policy Network)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 7)  # Assuming continuous action space without activation; adjust if needed
        )

    def forward(self, x):
        return self.model(x)
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
        pass

def move_to_joint_angles(joint_angles):
    for i, angle in enumerate(joint_angles):
        joints_publishers[i].publish(Float64(data=angle))
    rospy.sleep(0.01)

generator_path = 'best_generator.pth'
model = Generator().to(device)
model.load_state_dict(torch.load(generator_path))
model.eval()


def follow_trajectory(desired_trajectory):
    global end_effector_positions, record_position
    for point in desired_trajectory:
        point_3d = np.append(point, 0.5)  
        point_tensor = torch.tensor(point_3d, dtype=torch.float32).unsqueeze(0).to(device)
        joint_angles_tensor = model(point_tensor)
        joint_angles_np = joint_angles_tensor.cpu().detach().numpy().squeeze()
        # print(joint_angles_np)
        move_to_joint_angles(joint_angles_np)
        if record_position:
            update_end_effector_position()
            end_effector_positions.append(current_end_effector_position.copy())
        rospy.sleep(0.01)

if __name__ == "__main__":
    record_position = False
    initial_joint_angles = [0.0, -0.785398163, 0.0, -2.35619449, 0, 1.5707963267948966, 0.7853981633974483]
    rospy.sleep(2)
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)

    drawer = FreehandDrawer()

    trajectory = drawer.draw_trajectory()

    print(trajectory)

    record_position = True
    follow_trajectory(trajectory)
    record_position = False
    end_effector_positions = np.array(end_effector_positions)
    np.save('end_effector_positions.npy', end_effector_positions)
    animator = TrajectoryAnimator(end_effector_positions, trajectory)
    animator.animate()
