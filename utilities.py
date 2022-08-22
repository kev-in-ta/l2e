import numpy as np
import cv2

import tqdm

from typing import Tuple
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def get_mi(
    T: np.array,
    K: np.ndarray,
    data: dict,
    axis: bool = True,
) -> float:
    """Get MI score given images and pointclouds.

    Args:
        T: (6,) translations and rotation angles (ZYX)
        K: (3,3) matrix containing the projection
        data: dictionary of dicts with image and point cloud info
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
        H = param2hmatRodrigues(*T)
    else:
        H = param2hmat(*T)
        
    pbar = tqdm.tqdm(total=len(data))

    # loop through each scene in the dictionary
    for scene in data:

        # project points into image
        projected_points, tf_pc_int = points2image(
            data[scene]["pc_xyz"], data[scene]["pc_int"], K, H
        )

        # get scene histogram info
        hx, hy, hxy, n = calc_joint_hist(
            data[scene]["image"],
            projected_points[0:2, :],
            tf_pc_int,
            data[scene]["x_min"],
            data[scene]["x_max"],
            data[scene]["y_min"],
            data[scene]["y_max"],
        )

        # add to totals
        total_x = total_x + hx
        total_y = total_y + hy
        total_xy = total_xy + hxy
        total_n += n
        
        pbar.update(1)

    if total_n > 0:
        # attempt to smooth histogram with KDE
        p_x, p_y, p_xy = get_kde_convolve(total_x, total_y, total_xy)
        score = -calc_mi(p_x, p_y, p_xy)  # return the negative of MI for optimization
    else:
        score = 0
    
    print(f'MI Score: {score:.6f}')

    return score


def euler2mat(a: float = 0, b: float = 0, g: float = 0) -> np.ndarray:
    """Convert zyx Euler angles to 3x3 rotation matrix.

    Args:
        a: rotation about z axis in rad
        b: rotation about y axis in rad
        g: rotation about x axis in rad

    Returns:
        3x3 rotation matrix.
    """

    R = np.array(
        [
            [
                np.cos(a) * np.cos(b),
                np.cos(a) * np.sin(b) * np.sin(g) - np.sin(a) * np.cos(g),
                np.cos(a) * np.sin(b) * np.cos(g) + np.sin(a) * np.sin(g),
            ],
            [
                np.sin(a) * np.cos(b),
                np.sin(a) * np.sin(b) * np.sin(g) + np.cos(a) * np.cos(g),
                np.sin(a) * np.sin(b) * np.cos(g) - np.cos(a) * np.sin(g),
            ],
            [-np.sin(b), np.cos(b) * np.sin(g), np.cos(b) * np.cos(g)],
        ]
    )

    return R


def mat2euler(R: np.ndarray) -> np.array:
    """Convert 3x3 rotation matrix to ZYX Euler angles.

    Args:
        H: homogenous transformation.

    Returns:
        (3,) ZYX rotation angles
    """

    if np.abs(R[0, 0]) < 1e-8 and np.abs(R[1, 0]) < 1e-8:
        a = 0
        b = np.pi / 2
        g = np.arctan2(R[0, 1], R[1, 1])
    else:
        a = np.arctan2(R[1, 0], R[0, 0])
        b = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        g = np.arctan2(R[2, 1], R[2, 2])

    return np.array([a, b, g])


def param2hmat(
    x: float = 0, y: float = 0, z: float = 0, a: float = 0, b: float = 0, g: float = 0
) -> np.ndarray:
    """Convert xyz translation and zyx Euler angles to 4x4 homogenous matrix.

    Args:
        x: translation about x axis in m
        y: translation about y axis in m
        z: translation about z axis in m
        a: rotation about z axis in rad
        b: rotation about y axis in rad
        g: rotation about x axis in rad

    Returns:
        4x4 homogenous matrix.
    """

    H = np.zeros((4, 4))
    H[0:3, 0:3] = euler2mat(a, b, g)
    H[0, 3] = x
    H[1, 3] = y
    H[2, 3] = z
    H[3, 3] = 1

    return H


def param2hmatRodrigues(
    x: float = 0,
    y: float = 0,
    z: float = 0,
    r1: float = 0,
    r2: float = 0,
    r3: float = 0,
) -> np.ndarray:
    """Convert xyz translation and zyx Euler angles to 4x4 homogenous matrix.

    Args:
        x: translation about x axis in m
        y: translation about y axis in m
        z: translation about z axis in m
        r1: axis-angle component 1
        r2: axis-angle component 2
        r3: axis-angle component 3

    Returns:
        4x4 homogenous matrix.
    """

    H = np.zeros((4, 4))
    H[0:3, 0:3] = cv2.Rodrigues(np.array([r1, r2, r3]))[0]
    H[0, 3] = x
    H[1, 3] = y
    H[2, 3] = z
    H[3, 3] = 1

    return H


def hmat2param(
    H: np.ndarray,
) -> np.array:
    """Convert xyz translation and zyx Euler angles to 4x4 homogenous matrix.

    Args:
        H: (4,4) homogenous matrix.

    Returns:
        x: translation about x axis in m
        y: translation about y axis in m
        z: translation about z axis in m
        a: rotation about z axis in rad
        b: rotation about y axis in rad
        g: rotation about x axis in rad
    """

    T = H[0:3, 3]
    R = mat2euler(H[0:3, 0:3])

    P = np.hstack((T, R))

    return P


def points2image(pc_xyz: np.ndarray, pc_int: np.array, K: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project 3D points into 2D image space.

    Args:
        points_xyz: (n,3,) array containing xyz points
        points_int: (n,) array containing point intensities
        K: (3,3) matrix containing the projection
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
    tf_pc_xy = tf_pc_filt[0:3, :] / np.repeat(tf_pc_filt[2, :].reshape(1, -1), 3, axis=0)

    # project and return
    return K @ tf_pc_xy, tf_pc_int


def points2imageCV(
    pc_xyz: np.ndarray, pc_int: np.array, K: np.ndarray, D: np.ndarray, H: np.ndarray
) -> np.ndarray:
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
    pc_int: np.array,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> Tuple[np.array, np.array, np.ndarray, int]:
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
        h_x: (256,) count of LiDAR intensities for each value
        h_y: (256,) count of grayscale intensities for each point
        h_xy: (256,256,) count in joint histogram
        n: total number of points
    """

    h, w = image.shape

    roi_mask = (
        (pc_xy[1, :] < y_max)
        & (pc_xy[0, :] < x_max)
        & (pc_xy[1, :] > y_min)
        & (pc_xy[0, :] > x_min)
    )

    n = roi_mask.sum()

    im_coords = np.round(pc_xy[:, roi_mask]).astype(np.uint16)
    im_ints = image[im_coords[1, :], im_coords[0, :]]

    h_x, _ = np.histogram(pc_int[roi_mask].astype(int), bins=np.linspace(0, 256, 257))
    h_y, _ = np.histogram(im_ints, bins=np.linspace(0, 256, 257))
    h_xy, _, _ = np.histogram2d(
        pc_int[roi_mask].astype(int), im_ints, bins=np.linspace(0, 256, 257)
    )

    return h_x, h_y, h_xy, n


def calc_mi(c_x: np.array, c_y: np.array, c_xy: np.ndarray, n: int = 1) -> float:
    """Calculate mutual information criteria.

    Args:
        c_x: (256,) count of LiDAR intensities
        c_y: (256,) count of grayscale intensities
        c_xy: (256,256,) count of joint intensities
        n: number of points

    Returns:
        MI: mutual information criteria
    """

    if n == 0:
        return 0

    else:

        p_x = c_x / n
        p_y = c_y / n
        p_xy = c_xy / n

        H_x = np.sum(-p_x[p_x > 0] * np.log(p_x[p_x > 0]))
        H_y = np.sum(-p_y[p_y > 0] * np.log(p_y[p_y > 0]))
        H_xy = np.sum(-p_xy[p_xy > 0] * np.log(p_xy[p_xy > 0]))

        return H_x + H_y - H_xy


def get_kde_true(
    h_x: np.array,
    h_y: np.array,
    h_xy: np.array,
) -> Tuple[np.array, np.array, np.ndarray]:
    """Get Gaussian kernel density estimate entirely.

    Args:
        h_x: (256,) histogram of LiDAR intensities
        h_y: (256,) histogram of grayscale intensities
        h_xy: (256,256,) histogram of joint intensities

    Returns:
        p_kde_x: (256,) PDF for LiDAR intensities
        p_kde_y: (256,) PDF for image intensities
        p_kde_xy: (256,256,) joint PDF for join intensities
    """

    # get reconstructed point observations
    total_x_data = []
    for i, num in enumerate(total_x):
        total_x_data += [i] * int(num)

    total_y_data = []
    for i, num in enumerate(total_y):
        total_y_data += [i] * int(num)

    total_xy_data = []
    for i, y_data in enumerate(total_xy):
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

    X, Y = np.mgrid[0:255:256j, 0:255:256j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    total_kde_xy = gkde_xy.evaluate(positions)
    p_xy = total_kde_xy.T.reshape(X.shape)

    return p_x, p_y, p_xy


def get_kde_convolve(
    h_x: np.array,
    h_y: np.array,
    h_xy: np.array,
) -> Tuple[np.array, np.array, np.ndarray]:
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

    total_n = np.sum(h_x)

    mu_x = np.sum(h_x * np.linspace(0, 255, 256)) / total_n
    mu_y = np.sum(h_y * np.linspace(0, 255, 256)) / total_n

    sigma_x = np.sum(h_x * (np.linspace(0, 255, 256) - mu_x) ** 2) / total_n
    sigma_y = np.sum(h_y * (np.linspace(0, 255, 256) - mu_y) ** 2) / total_n

    factor = (total_n * (1 + 2) / 4) ** (-1 / (1 + 4))
    # factor2 = (total_n * (2 + 2) / 4)**(-1/(2+4))

    bw_x = factor * np.sqrt(sigma_x)
    bw_y = factor * np.sqrt(sigma_y)
    bw_xy = [factor * np.sqrt(sigma_x), factor * np.sqrt(sigma_y)]
    try:
        p_x = gaussian_filter1d(h_x / total_n, bw_x, mode="constant", cval=0.0)
        p_y = gaussian_filter1d(h_y / total_n, bw_y, mode="constant", cval=0.0)

        p_xy = gaussian_filter(h_xy / total_n, bw_xy, mode="constant", cval=0.0)
    except ValueError as e:
        print(e)
        p_x = np.zeros(256)
        p_y = np.zeros(256)
        p_xy = np.zeros((256, 256))

    return p_x, p_y, p_xy
