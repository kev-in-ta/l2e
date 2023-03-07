import datetime
import json
import pathlib
import time
from pathlib import Path
from typing import Tuple

import cv2
import fire
import numpy as np
from scipy.optimize import Bounds, minimize

from utilities import get_mi


def calibrate(
    image_path: str,
    lidar_path: str,
    output_path: str,
    bounded: bool = True,
    disp_bounds: float = 0.2,
    deg_bounds: float = 5.0,
    optimizer: str = "SLSQP",
    image_blur: float = 5,
    key: str = "cam",
    axis: bool = True,
) -> None:
    """Extrinsically calibrate camera to LiDAR via mutual information maximization.

    Args:
        image_path: path to camera folder
        lidar_path: path to lidar folder
        output_path: path to where the results are saved
        bounded: flag for optimizing with bounds on seed values
        disp_bounds: max displacement difference from seed values in m
        deg_bounds: max rotation difference from seed values in degrees
        optimizer: optimizer to use in {'nelder-mead', 'Powell', 'L-BFGS-B', 'BFGS', 'CG', 'SLSQP'}
        image_blur: gaussian blurring std on image to smooth optimization
        key: key for the intrinsic calibration in config.json
        axis: flag to use rotation vector over Euler angles
    """

    # convert paths
    image_path = Path(image_path)
    lidar_path = Path(lidar_path)
    output_path = Path(output_path)
    config_path = Path("config")

    # check paths
    if not image_path.exists():
        raise FileNotFoundError("Image directory does not exist!")
    if not lidar_path.exists():
        raise FileNotFoundError("LiDAR directory does not exist!")
    if not output_path.exists():
        output_path.mkdir()

    # load configuration
    K, D, tf_l2c = load_intrinsics(config_path / "config.json", key, axis)
    with open(config_path / "optimizer.json", "r", encoding="utf-8") as file_io:
        optim = json.load(file_io)

    # add bounds to optimixation
    if bounded and optimizer in ["nelder-mead", "Powell", "L-BFGS-B", "SLSQP"]:
        bound_error = np.array([disp_bounds] * 3 + [np.deg2rad(deg_bounds)] * 3)
        bounds = Bounds(tf_l2c - bound_error, tf_l2c + bound_error)
    else:
        bound_error = None

    # load scene data
    data = load_data(image_path, lidar_path, K, D, image_blur)

    print(f"Optimizing for {len(data)} scenes.")

    initial_mi = get_mi(tf_l2c, K, data, axis)

    # run optimizer
    start = time.time()
    res = minimize(
        get_mi,
        tf_l2c,
        (K, data, axis),
        method=optimizer,
        jac=optim[optimizer]["jac"],
        bounds=bounds,
        options=optim[optimizer]["options"],
    )
    end = time.time()

    # print optimizer results
    print(f"Time Elapsed: {end-start}s")
    print(f"Initial function value: {initial_mi:10.6f}")
    print(f"Optimal function value: {res.fun: 10.6f}")
    x, y, z, a, b, g = res.x
    print(f"X Translation: {x:8.4f} m")
    print(f"Y Translation: {y:8.4f} m")
    print(f"Z Translation: {z:8.4f} m")
    if axis:
        print(f"X Rotation: {a:11.4f} rad")
        print(f"Y Rotation: {b:11.4f} rad")
        print(f"Z Rotation: {g:11.4f} rad")
    else:
        print(f"Z Rotation: {np.rad2deg(a):11.4f} degrees")
        print(f"Y Rotation: {np.rad2deg(b):11.4f} degrees")
        print(f"X Rotation: {np.rad2deg(g):11.4f} degrees")

    # save results
    res_dict = {}
    res_dict["method"] = optimizer
    res_dict["jac"] = optim[optimizer]["jac"]
    res_dict["options"] = optim[optimizer]["options"]
    res_dict["fun"] = res.fun
    res_dict["smoothing"] = image_blur
    res_dict["x"] = res.x.tolist()
    res_dict["seed"] = tf_l2c.tolist()
    res_dict["bounded"] = bound_error.tolist()
    res_dict["time"] = end - start
    res_dict["scenes"] = list(data.keys())

    if axis:
        res_dict["representation"] = "axis-angle"
    else:
        res_dict["representation"] = "euler"

    output_file = (
        output_path / f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{key}.json'
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file_io:
        json.dump(res_dict, file_io, indent=4)


def load_data(
    image_folder: pathlib.Path,
    lidar_folder: pathlib.Path,
    K: np.ndarray,
    D: np.ndarray,
    image_blur: float,
) -> dict:
    """Load scene data into a dictionary for easy access.

    Args:
        image_folder: path to images
        lidar_folder: path to LiDAR scans
        K: (3,3) matrix containing the projection
        D: (5,) array containing distortion parameters
        image_blur: gaussian blurring std on image to smooth optimization
        raw16: flag to process raw16 Bayer images

    Returns:
        data: dictionary of categorized scenes with rectified images and scans.
    """
    data = {}

    images = image_folder.glob("*.png")
    image_stems = [image.stem for image in images]
    scans = lidar_folder.glob("*.txt")
    scan_stems = [scan.stem for scan in scans]

    matched_scenes = sorted([stem for stem in image_stems if stem in scan_stems])

    # add images and point cloud to scene dictionary
    for scene in matched_scenes:
        image_path = image_folder / f"{scene}.png"
        point_cloud = lidar_folder / f"{scene}.txt"

        # load image with any bit depth
        image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)

        # undistort image
        rect_image = cv2.undistort(image, K, D, None, K)

        # slightly blur images for optimization
        rect_image = cv2.GaussianBlur(rect_image, (0, 0), image_blur)

        h, w = rect_image.shape

        # load point cloud information
        pc = np.loadtxt(str(point_cloud), skiprows=0, delimiter=",")
        pc_xyz = pc[:, 0:3]
        pc_int = pc[:, 3]

        # save into dictionary
        data[scene] = {}
        data[scene]["image"] = rect_image
        data[scene]["pc_xyz"] = pc_xyz
        data[scene]["pc_int"] = pc_int
        data[scene]["x_min"] = 0
        data[scene]["x_max"] = w - 1
        data[scene]["y_min"] = 0
        data[scene]["y_max"] = h - 1

    return data


def load_intrinsics(
    intrinsic_file: pathlib.Path,
    frame: str = "cam",
    axis: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load intrinsic calibration values and seed extrinsics.

    Args:
        intrinsic_file: path to configuration json
        frame: dictionary key to calibration intrinsics

    Returns:
        K: (3,3) matrix containing the projection
        D: (5,) array containing distortion parameters
        T: (6,) translations and rotation angles (ZYX)
    """

    # load intrinsic parameters and seed extrinsics
    with open(intrinsic_file, "r", encoding="utf-8") as file_io:
        config = json.load(file_io)

    K = np.array(config[frame]["K"])
    D = np.array(config[frame]["D"])
    if axis:
        T = np.array(
            [
                config[frame]["x"],
                config[frame]["y"],
                config[frame]["z"],
            ]
            + config[frame]["Rodrigues"]
        )
    else:
        T = np.array(
            [
                config[frame]["x"],
                config[frame]["y"],
                config[frame]["z"],
                np.deg2rad(config[frame]["rz"]),
                np.deg2rad(config[frame]["ry"]),
                np.deg2rad(config[frame]["rx"]),
            ]
        )

    return K, D, T


if __name__ == "__main__":
    fire.Fire(calibrate)
