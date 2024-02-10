from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class TrajectoryAnimator:
    def __init__(self, actual_trajectory, reference_trajectory=None):
        # Exclude the first 50 values of the actual trajectory if it has enough points
        if len(actual_trajectory) > 50:
            self.actual_trajectory = np.array(actual_trajectory[50:])
        else:
            self.actual_trajectory = np.array(actual_trajectory)

        self.reference_trajectory = np.array(reference_trajectory) if reference_trajectory is not None else None

    def animate(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Setting up the plot limits based on trajectories
        if self.reference_trajectory is not None:
            all_points = np.concatenate((self.actual_trajectory, self.reference_trajectory), axis=0)
        else:
            all_points = self.actual_trajectory

        ax.set_xlim([all_points[:,0].min(), all_points[:,0].max()])
        ax.set_ylim([all_points[:,1].min(), all_points[:,1].max()])
        ax.set_zlim([all_points[:,2].min(), all_points[:,2].max()])

        # Plotting the reference trajectory if available
        if self.reference_trajectory is not None:
            ax.plot(self.reference_trajectory[:, 0], self.reference_trajectory[:, 1], self.reference_trajectory[:, 2], label='Reference Trajectory', linestyle='--', color='grey')

        line, = ax.plot([], [], [], 'r-', label='Actual Trajectory', lw=2)
        point, = ax.plot([], [], [], 'go')

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point

        def animate(i):
            line.set_data(self.actual_trajectory[:i, 0], self.actual_trajectory[:i, 1])
            line.set_3d_properties(self.actual_trajectory[:i, 2])
            point.set_data(self.actual_trajectory[i-1:i, 0], self.actual_trajectory[i-1:i, 1])
            point.set_3d_properties(self.actual_trajectory[i-1:i, 2])
            return line, point

        ani = animation.FuncAnimation(fig, animate, frames=len(self.actual_trajectory), init_func=init, blit=True, interval=30, repeat=False)
        
        plt.legend()
        plt.show()


