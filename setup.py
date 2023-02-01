#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

from setuptools import setup

setup(name='neuro_vsa', version='0.1', description='NVSA frontend and backend implementation',
      url='http://github.com/IBM/neuro-vsa', author='Michael Hersche', 
      packages=['neuro_vsa'], install_requires=['pytorch'], zip_safe=False)