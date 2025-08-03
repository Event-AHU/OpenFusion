# 用于对于物理模型进行建模并采样
# 修改了输入数据
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import time

def sample_interface_points(num_points, radius):
    theta = np.random.uniform(0, 2*np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    computed_radii = np.sqrt(x**2 + y**2)
    scale_factors = radius / computed_radii
    x = x * scale_factors
    y = y * scale_factors
    
    return x, y


def sample_annulus_points(num_points, r_inner, r_outer):
    theta = np.random.uniform(0, 2*np.pi, num_points)
    epsilon = 1e-10  
    r = np.sqrt(np.random.uniform((r_inner + epsilon)**2, (r_outer - epsilon)**2, num_points))
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def sample_disk_points(num_points, radius):
    theta = np.random.uniform(0, 2*np.pi, num_points)
    epsilon = 1e-10  
    r = np.sqrt(np.random.uniform(0, (radius- epsilon)**2 , num_points))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return x, y


def sample_rectangle_bottom_points(num_points, width, height):
    u = np.random.uniform(0, width, num_points)
    half_w = width/2
    half_h = height/2
    x = u - half_w
    y = np.full_like(u, -half_h)

    return np.array(x), np.array(y)


def sample_rectangle_left_points(num_points, width, height):
    u = np.random.uniform(0, height, num_points)
    half_w = width/2
    half_h = height/2
    x = np.full_like(u, -half_w)
    y = u - half_h

    return np.array(x), np.array(y)


def sample_rectangle_right_points(num_points, width, height):
    u = np.random.uniform(0, height, num_points)
    half_w = width/2
    half_h = height/2
    x = np.full_like(u, half_w)
    y = u - half_h

    return np.array(x), np.array(y)


def sample_rectangle_top_boundary_points(num_points, width, height):
    u = np.random.uniform(0, width, num_points)
    half_w = width/2
    half_h = height/2
    x = half_w - u
    y = np.full_like(u, half_h)

    return np.array(x), np.array(y)


def sample_rectangle_with_hole(num_points, width, height, hole_radius, center_x=None, center_y=None):
    if center_x is None:
        center_x = width / 2
    if center_y is None:
        center_y = height / 2
    rect_area = width * height
    hole_area = np.pi * hole_radius**2
    valid_area = rect_area - hole_area
    total_points = int(num_points * rect_area / valid_area * 1.1)  
    
    x = []
    y = []
    while len(x) < num_points:
        x_samples = np.random.uniform(-width/2, width/2, total_points)
        y_samples = np.random.uniform(-height/2, height/2, total_points)
        distances = np.sqrt(x_samples**2 + y_samples**2)
        valid_points = distances > hole_radius
        x_valid = x_samples[valid_points]
        y_valid = y_samples[valid_points]
        x.extend(x_valid[:num_points - len(x)])
        y.extend(y_valid[:num_points - len(y)])
    
    return np.array(x[:num_points]), np.array(y[:num_points])


def add_z_coordinate(points, z_min=-12, z_max=0):
    def sorted_random_times(size):
        return np.sort(np.random.uniform(1, 11, size))
    num_points = len(points[0])
    z = np.random.uniform(z_min, z_max, num_points)
    x_scaled = points[0] / 1
    y_scaled = points[1] / 1
    z_scaled = z / 1
    
    return np.column_stack((x_scaled, y_scaled, z_scaled, sorted_random_times(num_points)))


def add_time_only(points):
    def sorted_random_times(size):
        return np.sort(np.random.uniform(1, 11, size))
    num_points = len(points[0])
    x_scaled = points[0] / 1
    y_scaled = points[1] / 1
    
    return np.column_stack((x_scaled, y_scaled, sorted_random_times(num_points)))

def sample_rectangle_points(num_points, width, height, z_value, center_x=0, center_y=0):
    half_width = width / 2
    half_height = height / 2
    
    x = np.random.uniform(center_x - half_width, center_x + half_width, num_points)
    y = np.random.uniform(center_y - half_height, center_y + half_height, num_points)
    z = np.full(num_points, z_value)

    def sorted_random_times(size):
        return np.sort(np.random.uniform(1, 11, size))
    
    return np.column_stack((x, y, z, sorted_random_times(num_points)))

def plot_2d_sampling(all_points, labels):
    plt.figure(figsize=(12, 12))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'black']
    
    for (x, y), label, color in zip(all_points, labels, colors):
        plt.scatter(x, y, c=color, label=label, alpha=0.6, s=1)
    
    plt.title('2D Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig('pic/sampling_2d.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_sampling(all_3d_points, labels):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'black']
    
    for points, label, color in zip(all_3d_points, labels, colors):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]  # Extract x, y, z
        ax.scatter(x, y, z, c=color, label=label, alpha=0.6, s=1)
    
    ax.set_title('3D Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig('pic/sampling_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

def sample(seed):
    np.random.seed(int(seed))
    
    x_offset = 36.920
    y_offset = -1179.510
    
    multi = 5
    A1 = sample_disk_points(1, 6)
    A2 = sample_annulus_points(132*8*multi, 6, 7.5) 
    A3 = sample_annulus_points(344*3*multi, 7.5, 10.5)
    A4 = sample_rectangle_with_hole(1300*multi, 28, 30, 10.5)
    I1 = sample_interface_points(50*multi, 6) 
    I2 = sample_interface_points(52*multi, 7.5)
    I3 = sample_interface_points(58*multi, 10.5)
    R_bottom = sample_rectangle_bottom_points(50*multi, 28, 30)
    R_left = sample_rectangle_left_points(30*multi, 28, 30)
    R_right = sample_rectangle_right_points(30*multi, 28, 30)
    R_top = sample_rectangle_top_boundary_points(500*multi, 28, 30)
    R_front_3d = sample_rectangle_points(200*multi,28,30,0,center_x=36.920, center_y=-1179.510)
    R_back_3d = sample_rectangle_points(200*multi,28,30,-12,center_x=36.920, center_y=-1179.510)

    
    all_points = [A1, A2, A3, A4, I1, I2, I3, R_bottom, R_left, R_right, R_top,R_front_3d,R_back_3d]
    for i, points in enumerate(all_points):
        all_points[i] = (np.array(points[0]) + x_offset, 
                         np.array(points[1]) + y_offset)

    labels = ['A1', 'A2', 'A3', 'A4', 'I1', 'I2', 'I3', 'Bottom', 'Left', 'Right', 'Top']
    plot_2d_sampling(all_points, labels)
    sampling_data = {
    'A1': A1,
    'A2': A2,
    'A3': A3,
    'A4': A4,
    'I1': I1,
    'I2': I2,
    'I3': I3,
    'Bottom': R_bottom,
    'Left': R_left,
    'Right': R_right,
    'Top': R_top,
    }
    np.save('data/sampling_points_2d.npy', sampling_data)
    
    A1_3d = add_z_coordinate(all_points[0])
    A2_3d = add_z_coordinate(all_points[1])
    A3_3d = add_z_coordinate(all_points[2])
    A4_3d = add_z_coordinate(all_points[3])
    I1_3d = add_z_coordinate(all_points[4])
    I2_3d = add_z_coordinate(all_points[5])
    I3_3d = add_z_coordinate(all_points[6])
    R_bottom_3d = add_z_coordinate(all_points[7])
    R_left_3d = add_z_coordinate(all_points[8])
    R_right_3d = add_z_coordinate(all_points[9])
    R_top_3d = add_z_coordinate(all_points[10])

    scale_factor = 1
    
    for i in range(len(all_points)):
        all_points[i] = (all_points[i][0] * scale_factor, 
                        all_points[i][1] * scale_factor)
    
    all_3d_points = [A1_3d, A2_3d, A3_3d, A4_3d, I1_3d, I2_3d, I3_3d, 
                     R_bottom_3d, R_left_3d, R_right_3d, R_top_3d]
    
    for i in range(len(all_3d_points)):
        all_3d_points[i][:, :3] *= scale_factor
    
    A1_3d, A2_3d, A3_3d, A4_3d, I1_3d, I2_3d, I3_3d, R_bottom_3d, R_left_3d, R_right_3d, R_top_3d = all_3d_points

    sampling_data = {
        'A1': A1_3d,
        'A2': A2_3d,
        'A3': A3_3d,
        'A4': A4_3d,
        'I1': I1_3d,
        'I2': I2_3d,
        'I3': I3_3d,
        'Bottom': R_bottom_3d,
        'Left': R_left_3d,
        'Right': R_right_3d,
        'Top': R_top_3d,
        'Front': R_front_3d,
        'Back': R_back_3d
    }
    
    np.save('data/sampling_points_3d.npy', sampling_data)
    
    labels_3d = ['A1', 'A2', 'A3', 'A4', 'I1', 'I2', 'I3', 'Bottom', 'Left', 'Right', 'Top']
    plot_3d_sampling(all_3d_points, labels_3d)

if __name__ == "__main__":
    sample(1)