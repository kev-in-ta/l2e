"""Module containing utilities for geometric and statistical functions."""
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import gaussian_kde


@dataclass
class PointCloud:
    """Dataclass containing point cloud information."""

    xyz: np.ndarray
    intensity: np.ndarray


@dataclass
class PairedScene:
    """Dataclass containing paired scene information."""

    image: np.ndarray
    point_cloud: PointCloud
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def get_mi(
    tf_l2c: np.ndarray,
    camera_matrix: np.ndarray,
    data: Dict[str, PairedScene],
    axis: bool = True,
) -> float:
    """Get MI score given images and pointclouds.

    Args:
        tf_l2c: (6,) translations and rotation angles (ZYX / Rodrigues)
        camera_matrix: (3,3) matrix containing the projection
        data: dict of paired scene data with image and point cloud info
        axis: flag to use axis-angle representation

    Returns:
        MI: mutual information criteria
    """

    # initialize accumulated histogram arrays
    total_x = np.zeros(256)
    total_y = np.zeros(256)
    total_xy = np.zeros((256, 256))
    total_n = 0

    # get homogenous transformation matrix
    if axis:
        tf_l2c_matrix = convert_axis_vector_to_tf(*tf_l2c)
    else:
        tf_l2c_matrix = convert_euler_vector_to_tf(*tf_l2c)

    pbar = tqdm.tqdm(total=len(data))

    # loop through each scene in the dictionary
    for scene in data:
        # project points into image
        projected_points, tf_pc_int = project_points_to_image(
            data[scene].point_cloud.xyz, data[scene].point_cloud.intensity, camera_matrix, tf_l2c_matrix
        )

        # get scene histogram info
        h_x, h_y, h_xy, num_valid = calc_joint_hist(
            data[scene].image,
            projected_points[0:2, :],
            tf_pc_int,
            data[scene].x_min,
            data[scene].x_max,
            data[scene].y_min,
            data[scene].y_max,
        )

        # add to totals
        total_x = total_x + h_x
        total_y = total_y + h_y
        total_xy = total_xy + h_xy
        total_n += num_valid

        pbar.update(1)

    if total_n > 0:
        # attempt to smooth histogram with KDE
        p_x, p_y, p_xy = get_kde_convolve(total_x, total_y, total_xy)
        score = -calc_mi(p_x, p_y, p_xy)  # return the negative of MI for optimization
    else:
        score = 0

    print(f"MI Score: {score:.6f}")

    return score


def convert_euler_to_matrix(alpha: np.float64 = 0, beta: np.float64 = 0, gamma: np.float64 = 0) -> np.ndarray:
    """Convert zyx Euler angles to 3x3 rotation matrix.

    Args:
        alpha: rotation about z axis in rad
        beta: rotation about y axis in rad
        gamma: rotation about x axis in rad

    Returns:
        3x3 rotation matrix.
    """

    rotation_matrix = np.array(
        [
            [
                np.cos(alpha) * np.cos(beta),
                np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
                np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma),
            ],
            [
                np.sin(alpha) * np.cos(beta),
                np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
                np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma),
            ],
            [-np.sin(beta), np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma)],
        ]
    )

    return rotation_matrix


def convert_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to ZYX Euler angles.

    Args:
        rotation_matrix: homogenous transformation.

    Returns:
        (3,) ZYX rotation angles
    """

    if np.abs(rotation_matrix[0, 0]) < 1e-8 and np.abs(rotation_matrix[1, 0]) < 1e-8:
        alpha = 0
        beta = np.pi / 2
        gamma = np.arctan2(rotation_matrix[0, 1], rotation_matrix[1, 1])
    else:
        alpha = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        beta = np.arctan2(
            -rotation_matrix[2, 0],
            np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2),
        )
        gamma = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    return np.array([alpha, beta, gamma])


def convert_euler_vector_to_tf(
    t_x: np.float64 = 0,
    t_y: np.float64 = 0,
    t_z: np.float64 = 0,
    alpha: np.float64 = 0,
    beta: np.float64 = 0,
    gamma: np.float64 = 0,
) -> np.ndarray:
    """Convert xyz translation and zyx Euler angles to 4x4 homogenous matrix.

    Args:
        t_x: translation about x axis in m
        t_y: translation about y axis in m
        t_z: translation about z axis in m
        alpha: rotation about z axis in rad
        beta: rotation about y axis in rad
        gamma: rotation about x axis in rad

    Returns:
        4x4 homogenous matrix.
    """

    tf_matrix = np.zeros((4, 4))
    tf_matrix[0:3, 0:3] = convert_euler_to_matrix(alpha, beta, gamma)
    tf_matrix[0, 3] = t_x
    tf_matrix[1, 3] = t_y
    tf_matrix[2, 3] = t_z
    tf_matrix[3, 3] = 1

    return tf_matrix


def convert_axis_vector_to_tf(
    t_x: np.float64 = 0,
    t_y: np.float64 = 0,
    t_z: np.float64 = 0,
    r_x: np.float64 = 0,
    r_y: np.float64 = 0,
    r_z: np.float64 = 0,
) -> np.ndarray:
    """Convert xyz translation and axis-angle vector to 4x4 homogenous matrix.

    Args:
        t_x: translation about x axis in m
        t_y: translation about y axis in m
        t_z: translation about z axis in m
        r_x: axis-angle x-component
        r_y: axis-angle y-component
        r_z: axis-angle z-component

    Returns:
        4x4 homogenous matrix.
    """

    tf_matrix = np.zeros((4, 4))
    tf_matrix[0:3, 0:3] = cv2.Rodrigues(np.array([r_x, r_y, r_z]))[0]
    tf_matrix[0, 3] = t_x
    tf_matrix[1, 3] = t_y
    tf_matrix[2, 3] = t_z
    tf_matrix[3, 3] = 1

    return tf_matrix


def convert_tf_to_param(
    tf_matrix: np.ndarray,
) -> np.ndarray:
    """Convert 4x4 homogenous matrix to xyz translation and zyx Euler angles.

    Args:
        tf_matrix: (4,4) homogenous matrix.

    Returns:
        array consisting of:
        translation about x axis in m
        translation about y axis in m
        translation about z axis in m
        rotation about z axis in rad
        rotation about y axis in rad
        rotation about x axis in rad
    """

    t_vec = tf_matrix[0:3, 3]
    r_vec = convert_matrix_to_euler(tf_matrix[0:3, 0:3])

    p_vec = np.hstack((t_vec, r_vec))

    return p_vec


def project_points_to_image(
    pc_xyz: np.ndarray,
    pc_int: np.ndarray,
    camera_matrix: np.ndarray,
    tf_matrix: np.ndarray,
) -> np.ndarray:
    """Project 3D points into 2D image space.

    Args:
        points_xyz: (n,3,) array containing xyz points
        points_int: (n,) array containing point intensities
        camera_matrix: (3,3) matrix containing the projection
        tf_matrix: (4,4) matrix containing the transformation

    Returns:
        (m,2), (m,) filtered xy coordinates and intensity values
    """

    # augment point cloud with added 1 for matrix multiplication
    pc_extended = np.hstack((pc_xyz, np.ones(pc_int.reshape(-1, 1).shape)))

    # transform point cloud
    tf_pc = tf_matrix @ pc_extended.T

    # remove all points behind the projection
    forward_mask = tf_pc[2, :] > 0
    tf_pc_filt = tf_pc[:, forward_mask]
    tf_pc_int = pc_int[forward_mask]

    # scale by z-depth for projective geometry
    tf_pc_xy = tf_pc_filt[0:3, :] / np.repeat(tf_pc_filt[2, :].reshape(1, -1), 3, axis=0)

    # project and return
    return camera_matrix @ tf_pc_xy, tf_pc_int


def points2imageCV(pc_xyz: np.ndarray, pc_int: np.ndarray, K: np.ndarray, D: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project 3D points into 2D image space.

    Args:
        points_xyz: (n,3,) array containing xyz points
        points_int: (n,) array containing point intensities
        K: (3,3) matrix containing the projection
        D: (5,) array containing distortion parameters
        H: (4,4) matrix containing the transformation

    Returns:
        (m,2), (m,) filtered xy coordinates and intensity values
    """

    # augment point cloud with added 1 for matrix multiplication
    pc_extended = np.hstack((pc_xyz, np.ones(pc_int.reshape(-1, 1).shape)))

    # transform point cloud
    tf_pc = H @ pc_extended.T

    # remove all points behind the projection
    forward_mask = tf_pc[2, :] > 0
    tf_pc_filt = tf_pc[:, forward_mask]
    tf_pc_int = pc_int[forward_mask]

    # scale by z-depth for projective geometry
    tf_pc_xy, _ = cv2.projectPoints(tf_pc_filt[0:3, :], np.eye(3), np.zeros((3, 1)), K, D)

    # project and return
    return tf_pc_xy.reshape(-1, 2).T, tf_pc_int


def calc_joint_hist(
    image: np.ndarray,
    pc_xy: np.ndarray,
    pc_int: np.ndarray,
    x_min: np.int64,
    x_max: np.int64,
    y_min: np.int64,
    y_max: np.int64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Generate joint histogram.

    Args:
        image: (w,h) grayscale image
        pc_xy: (m,2) point cloud positions in the rectified frame
        pc_int: (m,) LiDAR intensity values
        x_min: the left hand bound of ROI
        x_max: the right hand bound of ROI
        y_min: the bottom bound of ROI
        y_max: the top bound of ROI

    Returns:
        hist_x: (256,) count of LiDAR intensities for each value
        hist_y: (256,) count of grayscale intensities for each point
        hist_xy: (256,256,) count in joint histogram
        num_valid: total number of valid points
    """

    roi_mask = (pc_xy[1, :] < y_max) & (pc_xy[0, :] < x_max) & (pc_xy[1, :] > y_min) & (pc_xy[0, :] > x_min)

    num_valid = roi_mask.sum()

    im_coords = np.round(pc_xy[:, roi_mask]).astype(np.uint16)
    im_ints = image[im_coords[1, :], im_coords[0, :]]

    hist_x, _ = np.histogram(pc_int[roi_mask].astype(int), bins=np.linspace(0, 256, 257))
    hist_y, _ = np.histogram(im_ints, bins=np.linspace(0, 256, 257))
    hist_xy, _, _ = np.histogram2d(pc_int[roi_mask].astype(int), im_ints, bins=np.linspace(0, 256, 257))

    return hist_x, hist_y, hist_xy, num_valid


def calc_mi(c_x: np.ndarray, c_y: np.ndarray, c_xy: np.ndarray, num_points: np.int64 = 1) -> np.float64:
    """Calculate mutual information criteria.

    Args:
        c_x: (256,) count of LiDAR intensities
        c_y: (256,) count of grayscale intensities
        c_xy: (256,256,) count of joint intensities
        num_points: number of points

    Returns:
        MI: mutual information criteria
    """

    if num_points == 0:
        return 0

    else:
        p_x = c_x / num_points
        p_y = c_y / num_points
        p_xy = c_xy / num_points

        entropy_x = np.sum(-p_x[p_x > 0] * np.log(p_x[p_x > 0]))
        entropy_y = np.sum(-p_y[p_y > 0] * np.log(p_y[p_y > 0]))
        entropy_xy = np.sum(-p_xy[p_xy > 0] * np.log(p_xy[p_xy > 0]))

        return entropy_x + entropy_y - entropy_xy


def get_kde_true(
    hist_x: np.ndarray,
    hist_y: np.ndarray,
    hist_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get Gaussian kernel density estimate entirely.

    Args:
        hist_x: (256,) histogram of LiDAR intensities
        hist_y: (256,) histogram of grayscale intensities
        hist_xy: (256,256,) histogram of joint intensities

    Returns:
        p_kde_x: (256,) PDF for LiDAR intensities
        p_kde_y: (256,) PDF for image intensities
        p_kde_xy: (256,256,) joint PDF for join intensities
    """

    # get reconstructed point observations
    total_x_data = []
    for i, num in enumerate(hist_x):
        total_x_data += [i] * int(num)

    total_y_data = []
    for i, num in enumerate(hist_y):
        total_y_data += [i] * int(num)

    total_xy_data = []
    for i, y_data in enumerate(hist_xy):
        for j, num in enumerate(y_data):
            total_xy_data += [(i, j)] * int(num)
    total_xy_data = np.array(total_xy_data).T

    # initialize KDE
    gkde_x = gaussian_kde(total_x_data, bw_method="silverman")
    gkde_y = gaussian_kde(total_y_data, bw_method="silverman")
    gkde_xy = gaussian_kde(total_xy_data, bw_method="silverman")

    # get evaluation points and retrieve probabilities
    x_pts = np.linspace(0, 255, 256)
    p_x = gkde_x.evaluate(x_pts)

    y_pts = np.linspace(0, 255, 256)
    p_y = gkde_y.evaluate(y_pts)

    x_grid, y_grid = np.mgrid[0:255:256, 0:255:256]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    total_kde_xy = gkde_xy.evaluate(positions)
    p_xy = total_kde_xy.T.reshape(x_grid.shape)

    return p_x, p_y, p_xy


def get_kde_convolve(
    hist_x: np.ndarray,
    hist_y: np.ndarray,
    hist_xy: np.ndarray,
    use_silverman_1d: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get Gaussian kernel density estimate through convolution.

    Args:
        h_x: (256,) histogram of LiDAR intensities
        h_y: (256,) histogram of grayscale intensities
        h_xy: (256,256,) histogram of joint intensities

    Returns:
        p_kde_x: (256,) PDF for LiDAR intensities
        p_kde_y: (256,) PDF for image intensities
        p_kde_xy: (256,256,) joint PDF for join intensities
    """

    total_n = np.sum(hist_x)

    mu_x = np.sum(hist_x * np.linspace(0, 255, 256)) / total_n
    mu_y = np.sum(hist_y * np.linspace(0, 255, 256)) / total_n

    sigma_x = np.sum(hist_x * (np.linspace(0, 255, 256) - mu_x) ** 2) / total_n
    sigma_y = np.sum(hist_y * (np.linspace(0, 255, 256) - mu_y) ** 2) / total_n

    if use_silverman_1d:
        factor = (total_n * (1 + 2) / 4) ** (-1 / (1 + 4))
    else:
        factor = (total_n * (2 + 2) / 4) ** (-1 / (2 + 4))

    bw_x = factor * np.sqrt(sigma_x)
    bw_y = factor * np.sqrt(sigma_y)
    bw_xy = [factor * np.sqrt(sigma_x), factor * np.sqrt(sigma_y)]
    try:
        p_x = gaussian_filter1d(hist_x / total_n, bw_x, mode="constant", cval=0.0)
        p_y = gaussian_filter1d(hist_y / total_n, bw_y, mode="constant", cval=0.0)

        p_xy = gaussian_filter(hist_xy / total_n, bw_xy, mode="constant", cval=0.0)
    except ValueError as err:
        print(err)
        p_x = np.zeros(256)
        p_y = np.zeros(256)
        p_xy = np.zeros((256, 256))

    return p_x, p_y, p_xy
