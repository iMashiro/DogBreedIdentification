from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='It can merge two dogs',
    author='britohyago',
    license='',
    install_requires=['pandas',
                        'numpy',
                        'matplotlib',
                        'torch',
                        'torchvision']
)