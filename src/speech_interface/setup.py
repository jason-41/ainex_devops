from setuptools import find_packages, setup

package_name = 'speech_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wbo',
    maintainer_email='wbo1421@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'speech_to_text = speech_interface.speech_to_text_node:main',
            'text_to_speech = speech_interface.text_to_speech_node:main',
        ],
    },
)
