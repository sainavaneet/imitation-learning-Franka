import numpy as np

class CircleTraj:
    def __init__(self, center_x=0.5, center_y=0, center_z=0, radius=0.1, points=100):

        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.radius = radius
        self.points = points

    def generate_trajectory(self):

        theta = np.linspace(0, 2 * np.pi, self.points)
        y = self.center_y + self.radius * np.cos(theta)
        z = self.center_z + self.radius * np.sin(theta)
        x = np.full_like(y, self.center_x)
        
        return [np.array([x[i], y[i], z[i]]) for i in range(len(x))]
