#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

from setuptools import setup

setup(name='nvsa', version='0.1', description='NVSA frontend and backend implementation',
      url='http://github.com/IBM/neuro-vector-symbolic-architectures', author='Michael Hersche', 
      packages=['nvsa'], install_requires=['pytorch'], zip_safe=False)
