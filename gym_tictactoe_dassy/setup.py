import setuptools
from setuptools import setup

setup(name='gym_tictactoe_dassy',
      description='Implementation of an OpenAI Gym learning environment of a simple tic tac toe game.',
      version='0.0.1',
      author='Laure Dassy',
      url="https://github.com/LaureDas/gym_tictactoe_dassy.git",
      packages=setuptools.find_packages(),
      install_requires=['gym'])