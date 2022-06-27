
from setuptools import setup

import progress


setup(
    name='progress',
    version=progress.__version__,
    description='Easy to use progress bars',
    long_description=open('README.rst').read(),
    author='None',
    author_email='None',
    url='None',
    license='None',
    packages=['progress'],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: None',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
