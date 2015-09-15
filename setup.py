#!/usr/bin/env python

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import numpy
setup(
    name='scaling',
    version='0.1',
    packages=['scaling','scaling.powerlaw'],
    install_requires=['numpy','scipy'],
    author='John Walk',
    author_email='jrwalk@mit.edu',
    url='https://github.com/jrwalk/scaling',
    description='Tools for scaling-law development',
    long_description=open('README.md','r').read(),
    license='GPL'
)
