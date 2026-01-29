from setuptools import setup

package_name = 'teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hrs2025',
    maintainer_email='nivilian007@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crouch = teleop.crouch:main',
            'hands_control = teleop.hands_control:main',
            'main_control = teleop.main_control:main',
            'turn_around = teleop.turn_around:main',
            'walk_to_aruco = teleop.walk_to_Aruco:main',
            'grasp = teleop.grasp:main',
        ],
    },
)
