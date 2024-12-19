# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='#IDP30| DyNaMo | Measuring lipid bilayers height with accuracy',
    version='1.0.0',
    description='Python scripts for AFM image processing and analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Thang Duong Quoc LE',
    author_email='duong-quoc-thang.le@univ-amu.fr',
    classifiers=[
        'Development Status :: 1 - Production/Stable',
        'Intended Audience :: Science/Research',
    ],
    keywords='atomic force microscopy, afm, image processing',
    python_requires='>=3.9,<=3.11',
    packages=find_packages(exclude=('outputs')),
    install_requires=['matplotlib==3.8.2',
                      'numpy==1.26.4',
                      'pandas==2.1.4',
                      'Pillow==10.2.0',
                      'scikit-image==0.22.0',
                      'scikit-learn==1.3.0',
                      'scipy==1.11.4',
                      'tqdm==4.65.0'],
    license='MIT',
    project_urls={  # Optional
        'Source': 'https://centuri-livingsystems.org/multi-engineering-platform/project-list/',
                },
    )