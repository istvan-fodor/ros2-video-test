from setuptools import find_packages, setup
import os
import glob

package_name = 'analog_gauge_reader'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/analog_gauge_reader.launch.py'])
    ],
    install_requires=['setuptools', 'ultralytics', 'mmcv', 'mmocr', 'scikit-learn'],
    zip_safe=True,
    maintainer='istvan.fodor',
    maintainer_email='info@istvanfodor.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gauge_reader_node = analog_gauge_reader.analog_gauge_reader:main',
        ],
    },
)
