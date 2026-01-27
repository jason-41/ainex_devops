from setuptools import setup, find_packages
from glob import glob
import os

package_name = "vision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hrs2025",
    maintainer_email="nivilian007@gmail.com",
    description="Vision package",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "aruco_detector_node = vision.aruco_detector_node:main",
            # "dl_node = vision.dl_node:main",
            "object_detector = vision.object_detector:main",
            "target_object_detector_node = vision.target_object_detector_node:main",
            "tf_debug_publisher_node = vision.tf_debug_publisher_node:main",
            "undistortion = vision.undistortion:main",
            # "capture_node = vision.capture_node:main",
        ],
    },
)
