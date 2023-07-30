"""Command line tool for extracting a rosbag."""
import pathlib
import time
from pathlib import Path
from typing import List

import cv2
import fire
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
import tqdm


def extract_bag(
    bag_dir: str,
    output_dir: str,
    event: bool = False,
    lidar: bool = False,
) -> None:
    """Extract rosbags to single image frames and pointclouds.

    Args:
        bag_file: path to directory containing rosbags for extraction
        output_dir: path to directory where rosbags are extracted to
        event: set if event image frame should be extracted
        lidar: set if lidar point cloud should be extracted
    """

    bag_path = Path(bag_dir)
    output_path = Path(output_dir)

    if not bag_path.exists():
        raise FileNotFoundError("Bag directory does not exist!")
    if not output_path.exists():
        output_path.mkdir()
        (output_path / "event").mkdir()
        (output_path / "lidar").mkdir()

    bag_files: List[Path] = []
    bag_files += bag_path.glob("*.bag")

    start = time.time()

    for bag_file in bag_files:
        bag = rosbag.Bag(bag_file)
        print(output_path / bag_file.stem)
        if event:
            extract_event(bag, output_path / "event" / f"{bag_file.stem}.png")
        if lidar:
            extract_lidar(bag, output_path / "lidar" / f"{bag_file.stem}.txt")

    bag.close()

    end = time.time()

    print(f"{len(bag_files)} scenes completed in {end-start:.3f}s")


def extract_event(
    bag: rosbag.bag.Bag,
    output_name: pathlib.Path,
    event_limit: int = 10000,
    clip_count: int = 127,
) -> None:
    """Extract single event image from rosbag.

    Args:
        bag: rosbag with topics to parse
        output_name: file name to save image as
        event_limit: number of events at which to stop accumulating
        clip_count: maximum intensity value to clip results to
    """

    msg = None

    # generate empty array
    image = np.zeros((720, 1280), dtype=np.uint16)

    total = int(
        min(
            int(bag.get_message_count("/prophesee/camera/cd_events_buffer")),
            event_limit,
        )
    )
    pbar = tqdm.tqdm(total=total)

    # accumulate events up to the limit
    event_batch = 0
    for _, msg, _ in bag.read_messages(topics=["/prophesee/camera/cd_events_buffer"]):
        for event in msg.events:
            image[event.y, event.x] += 1
        event_batch += 1
        pbar.update(1)
        if event_batch >= total:
            break

    if msg:
        # clip accumulated events to prevent scaling issues
        image[image > clip_count] = clip_count
        cv2.imwrite(str(output_name), image.astype(np.uint8))


def extract_lidar(bag: rosbag.bag.Bag, output_name: pathlib.Path) -> None:
    """Extract single LiDAR scan from rosbag and filter non-returns.

    Args:
        bag: rosbag with topics to parse
        output_name: file name to save point cloud as
    """

    msg = None

    # read first point cloud
    for _, msg, _ in bag.read_messages(topics=["/rslidar_points"]):
        break

    if msg:
        # generate point cloud data
        sample_pc = pc2.read_points_list(msg)

        # filter out non-returns
        points = np.array(sample_pc)
        filter_nan = ~(points[:, 3] == 0)
        points_clean = points[filter_nan, :]

        np.savetxt(str(output_name), points_clean, delimiter=",")


if __name__ == "__main__":
    fire.Fire(extract_bag)
