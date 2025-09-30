from setuptools import setup, find_packages

packages = find_packages(include=[
    'combined_action_pillar_core',
    'combined_action_pillar_core.*',
    'gazelle',
    'gazelle.*',
])

setup(
    name='combined_action_pillar_core',
    version='0.1.0',
    packages=packages,
    package_data={
        'combined_action_pillar_core': ['assets/models/*'],
        'gazelle': ['*.pt'],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/combined_action_pillar_core']),
        ('share/combined_action_pillar_core', ['package.xml']),
    ],
    install_requires=[
        'opencv-python',
        'numpy',
        'joblib',
        'mediapipe',
        'ultralytics',
        'torch'
    ],
    author='PILLAR WP2 Team',
    author_email='info@pillar-project.eu',
    maintainer='PILLAR WP2 Team',
    maintainer_email='info@pillar-project.eu',
    description='Combined action perception core library for the PILLAR project.',
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    zip_safe=False,
)
