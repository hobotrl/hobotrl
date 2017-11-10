from setuptools import setup, find_packages
import sys

setup(name='hobotrl',
      packages=[package for package in find_packages()
                if package.startswith('hobotrl')],
      install_requires=[
          'gym>=0.9.1[all]',
          'scipy',
          'numpy>=1.10.4',
          # 'tensorflow >= 1.0.0',
          'httplib2',
          'wrapt'
      ],
      description="Reinforcement Learning Algorithm libraries and experiment collections from Hobot RLer",
      author="Hobot RLer",
      url='https://github.com/hobotrl/hobotrl',
      author_email="",
      version="0.1")
