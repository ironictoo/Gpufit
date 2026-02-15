"""
setup script for pyCpufit
"""

from setuptools import setup, find_packages
import os
import sys
from io import open
import pycpufit.version as vs

if os.name == 'nt':
    lib_ext = '.dll'
elif os.name == 'posix':
    lib_ext = '.dylib' if sys.platform == 'darwin' else '.so'
else:
    raise RuntimeError('OS {} not supported'.format(os.name))

HERE = os.path.abspath(os.path.dirname(__file__))


def get_long_description():
    with open(os.path.join(HERE, 'README.txt'), encoding='utf-8') as f:
        return f.read()


if __name__ == "__main__":
    setup(
        name='pyCpufit',
        version=vs.__version__,
        description='Levenberg Marquardt curve fitting on CPU',
        long_description=get_long_description(),
        url='https://github.com/gpufit/Gpufit',
        author='M. Bates, A. Przybylski, B. Thiel, and J. Keller-Findeisen',
        author_email='a@b.c',
        license='MIT license',
        classifiers=[],
        keywords='Levenberg Marquardt, curve fitting, CPU',
        packages=find_packages(where=HERE),
        package_data={'pycpufit': ['*{}'.format(lib_ext)]},
        install_requires=['NumPy>=1.0'],
        zip_safe=False,
    )
