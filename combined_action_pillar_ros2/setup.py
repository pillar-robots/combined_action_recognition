from setuptools import setup, find_packages
from glob import glob

package_name = 'combined_action_pillar_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'combined_action_pillar_core'],
    zip_safe=True,
    author='PILLAR WP2 Team',
    author_email='info@pillar-project.eu',
    maintainer='PILLAR WP2 Team',
    maintainer_email='info@pillar-project.eu',
    description='ROS 2 integration for the Combined Action PILLAR module.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'combined_action_node = combined_action_pillar_ros2.nodes.combined_action_node:main',
        ],
    },
)
