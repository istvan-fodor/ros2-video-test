from setuptools import find_packages, setup

package_name = 'fusiongrid_robot'

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
    maintainer='istvan.fodor',
    maintainer_email='info@istvanfodor.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_producer = fusiongrid_robot.camera_producer:main',
            'grounding_dino_node = fusiongrid_robot.grounding_dino_node:main',
            'speech_recognition_node = fusiongrid_robot.speech_recognition_node:main',
            'segmentation_node = fusiongrid_robot.segmentation_node:main',
            'video_publisher = fusiongrid_robot.video_publisher:main'
        ],
    },
)
