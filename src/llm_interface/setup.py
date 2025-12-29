from setuptools import find_packages, setup

package_name = 'llm_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/llm_interface']),
        ('share/llm_interface', ['package.xml']),
        ('share/llm_interface/launch', ['launch/llm_with_auth.launch.py']),
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
            'llm_server_node = llm_interface.llm_server_node:main',
            'llm_cli_node = llm_interface.llm_cli_node:main',
        ],
    },
)
