# import rospy
# import numpy as np
# import json
# from std_msgs.msg import Float64
# from sensor_msgs.msg import JointState
# import tf2_ros
# from kineamatics.panda_kinematics import PandaWithHandKinematics

# current_joint_angles = None
# current_end_effector_position = None
# dataset = []
# joints_publishers = [rospy.Publisher(f'/franka/panda_joint{i+1}_controller/command', Float64, queue_size=10) for i in range(7)]

# def joint_states_callback(msg):
#     global current_joint_angles
#     current_joint_angles = np.array(msg.position[:7])

# def update_end_effector_position(tf_buffer):
#     global current_end_effector_position
#     try:
#         trans = tf_buffer.lookup_transform('panda_link0', 'panda_hand', rospy.Time(0), rospy.Duration(1.0))
#         current_end_effector_position = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
#     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#         rospy.logwarn("TF lookup failed: %s", e)

# def publish_joint_positions(joint_positions):
#     for i, position in enumerate(joint_positions):
#         joints_publishers[i].publish(Float64(position))
#     rospy.sleep(0.5)  # Adjust sleep time as needed for your setup

# def main():
#     rospy.init_node('trajectory_controller')
#     initial_joint_angles = [0, -0.7, 0, -2.35619449, 0, 1.57079632679, 0.785398163397]
#     publish_joint_positions(initial_joint_angles)
#     rospy.sleep(2)  # Ensure the robot has reached the initial position
#     publish_joint_positions(initial_joint_angles)
#     rospy.sleep(2)  # Ensure the robot has reached the initial position



#     tf_buffer = tf2_ros.Buffer()
#     tf_listener = tf2_ros.TransformListener(tf_buffer)
#     rospy.Subscriber('/franka/joint_states', JointState, joint_states_callback)

#     kinematics = PandaWithHandKinematics()
#     waypoints_square = np.array([[0.6, 0.2, 0.6],[0.6, 0.2, 0.6], [0.6, 0.0, 0.6], [0.4, 0.0, 0.6], [0.4, 0.2, 0.6], [0.6, 0.2, 0.6]])
#     orientation_quat = np.array([0.0, 1.0, 0.0, 0.0])

#     for waypoint in waypoints_square:
#         rospy.sleep(0.1)  # Short delay to ensure current_joint_angles is updated
#         current_state_before_movement = current_joint_angles.copy() if current_joint_angles is not None else None

#         solution = kinematics.ik(current_joint_angles, waypoint.astype(np.float64), orientation_quat)
#         if solution is not None and not np.isnan(solution).any():
#             publish_joint_positions(solution)
#             rospy.sleep(1)  # Wait for movement to complete and stabilize
#             update_end_effector_position(tf_buffer)

#             dataset_entry = {
#                 "current_state": current_state_before_movement.tolist() if current_state_before_movement is not None else None,
#                 "current_end_effector_position": current_end_effector_position.tolist(),
#                 "joint_angles": current_joint_angles.tolist() if current_joint_angles is not None else None
#             }
#             dataset.append(dataset_entry)
#         else:
#             rospy.logwarn("IK solution not found or invalid for waypoint: %s", waypoint)

#     with open("/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/datasets/robotics_trajectory_dataset.json", "w") as json_file:
#         json.dump(dataset, json_file)
#     rospy.loginfo("Dataset saved to datasets/robotics_trajectory_dataset.json")

# if __name__ == '__main__':
#     main()



import rospy
import numpy as np
import json
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import tf2_ros
from kineamatics.panda_kinematics import PandaWithHandKinematics

current_joint_angles = None
current_end_effector_position = None
dataset = []
joints_publishers = [rospy.Publisher(f'/franka/panda_joint{i+1}_controller/command', Float64, queue_size=10) for i in range(7)]

def joint_states_callback(msg):
    global current_joint_angles
    current_joint_angles = np.array(msg.position[:7])

def update_end_effector_position(tf_buffer):
    global current_end_effector_position
    try:
        trans = tf_buffer.lookup_transform('panda_link0', 'panda_hand', rospy.Time(0), rospy.Duration(1.0))
        current_end_effector_position = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn("TF lookup failed: %s", e)

def publish_joint_positions(joint_positions):
    joints_publishers = [rospy.Publisher(f'/franka/panda_joint{i+1}_controller/command', Float64, queue_size=10) for i in range(7)]
    for i, position in enumerate(joint_positions):
        joints_publishers[i].publish(Float64(position))
    rospy.sleep(1)  # Adjust sleep time as needed for your setup

def move_to_joint_angles(joint_angles):
    for i, angle in enumerate(joint_angles):
        joints_publishers[i].publish(Float64(data=angle))
    rospy.sleep(0.03)

def main():
    rospy.init_node('trajectory_controller')
    initial_joint_angles = [0, -0.7, 0, -2.35619449, 0, 1.57079632679, 0.785398163397]
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)
    move_to_joint_angles(initial_joint_angles)
    rospy.sleep(2)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber('/franka/joint_states', JointState, joint_states_callback)

    kinematics = PandaWithHandKinematics()
    waypoints_square = np.array([[0.6, 0.2, 0.6],[0.6, 0.2, 0.6], [0.6, 0.0, 0.6], [0.4, 0.0, 0.6], [0.4, 0.2, 0.6], [0.6, 0.2, 0.6]])  # Ensure waypoints are numpy arrays
    orientation_quat = np.array([1.0, 1.0, 0.0, 0.0])  # Confirm orientation is a numpy array with shape (4,)

    for waypoint in waypoints_square:


        while current_joint_angles is None:
            rospy.sleep(2)  # Wait for the current joint angles to be available

        # Ensure waypoint is a numpy array with the correct shape when passed
        solution = kinematics.ik(current_joint_angles, waypoint.astype(np.float64), orientation_quat)
        if solution is not None and not np.isnan(solution).any():
            publish_joint_positions(solution)
            rospy.sleep(0.01)  # Wait for movement to complete and stabilize
            update_end_effector_position(tf_buffer)
            if current_joint_angles is not None and current_end_effector_position is not None:
                dataset_entry = {
                    "current_joint_angles": current_joint_angles.tolist(),
                    "current_end_effector_position": current_end_effector_position.tolist()
                }
                dataset.append(dataset_entry)
        else:
            rospy.logwarn("IK solution not found or invalid for waypoint: %s", waypoint)

    with open("/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/datasets/robotics_trajectory_dataset.json", "w") as json_file:
        json.dump(dataset, json_file)
    rospy.loginfo("Dataset saved to datasets/robotics_trajectory_dataset.json")

if __name__ == '__main__':
    main()
