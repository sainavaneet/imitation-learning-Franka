import matplotlib.pyplot as plt
import numpy as np
from kineamatics.panda_kinematics import PandaWithHandKinematics
import json

class FreehandDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-0.3, 0.3)
        self.ax.set_xlim(0.2, 0.6)
        self.ax.spines['left'].set_position(('data', 0))
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        self.ax.axhline(0, color='black')
        self.ax.axvline(0, color='black')
        self.line, = self.ax.plot([], [], 'r-')
        self.xs = []
        self.ys = []
        self.is_drawing = False

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.is_drawing = True
        self.xs = [event.xdata]
        self.ys = [event.ydata]

    def on_release(self, event):
        self.is_drawing = False

    def on_move(self, event):
        if not self.is_drawing or event.inaxes != self.ax:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        plt.draw()

    def draw_trajectory(self):
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmove = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        plt.title('Close window when done.')
        plt.show()
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmove)
        return list(zip(self.xs, self.ys))
    
    def interpolate_trajectory(self, num_points=10):
        if len(self.xs) < 2:  # Need at least two points to interpolate
            return list(zip(self.xs, self.ys))

        new_xs = []
        new_ys = []
        for i in range(len(self.xs) - 1):
            x_values = np.linspace(self.xs[i], self.xs[i + 1], num_points)
            y_values = np.linspace(self.ys[i], self.ys[i + 1], num_points)
            new_xs.extend(x_values)
            new_ys.extend(y_values)

        new_xs.append(self.xs[-1])
        new_ys.append(self.ys[-1])

        return list(zip(new_xs, new_ys))

if __name__ == "__main__":
    drawer = FreehandDrawer()
    trajectory = drawer.draw_trajectory()
    interpolated_trajectory = drawer.interpolate_trajectory(num_points=10) 
 
    dataset = []
    calculate_ik = PandaWithHandKinematics()

    for point in interpolated_trajectory:
        desired_position = np.array([point[0], point[1], 0.5])
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

    file_name = "datasets/newdataset.json"
    with open(file_name, "w") as json_file:
        json.dump(dataset, json_file)

    print("Dataset saved to", file_name)