from setuptools import find_packages, setup
package_name = 'ainex_vision'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages=['ainex_vision'],
    package_dir={'ainex_vision': 'ainex_vision'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wenlan Shen',
    maintainer_email='wenlan.shen@tum.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_sub = ainex_vision.camera_sub:main',
            'face_detection_node = ainex_vision.face_detection_node:main',
            'aruco_tf_broadcaster = ainex_vision.aruco_tf_broadcaster:main',
            'aruco_detection_node = ainex_vision.aruco_detection_node:main',
            'aruco_marker_node = ainex_vision.aruco_marker_node:main',
        ],
    },
)
