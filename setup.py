from pathlib import Path
from setuptools import setup

description = ['Training and evaluation of the ICCV 2025 paper PixCuboid: Room Layout Estimation from Multi-view Featuremetric Alignment']

with open(str(Path(__file__).parent / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

with open(str(Path(__file__).parent / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

extra_dependencies = ['jupyter', 'scikit-learn', 'ffmpeg-python', 'kornia', 'rerun-sdk', 'rerun-sdk[notebook]']

deeplsd_dependencies = ['flow_vis', 'kornia>=0.6', 'scikit-image', 'seaborn', 'deeplsd @ git+https://github.com/cvg/DeepLSD.git', 'pytlsd @ git+https://github.com/iago-suarez/pytlsd.git']

setup(
    name='PixCuboid',
    version='1.0',
    packages=['pixloc'],
    python_requires='>=3.6',
    install_requires=dependencies,
    extras_require={'extra': extra_dependencies, 'deeplsd': deeplsd_dependencies},
    author='Paul-Edouard Sarlin (PixLoc), Gustav Hanning (PixCuboid)',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/ghanning/PixCuboid/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
