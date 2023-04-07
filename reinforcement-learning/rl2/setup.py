import importlib

from setuptools import setup, find_packages

assert importlib.util.find_spec('torch'), \
    'manually pre-install proper torch version for your system'
extras = {
    'gnn': ['torch_sparse', 'torch_scatter', 'torch_geometric']
}

setup(
    name='rl2',
    version='1.0.0',
    url='https://github.com/kc-ml2/rl2',
    author='Anthony Jung',
    author_email='jwseok0727@gmail.com',
    packages=find_packages(),
    install_requires=[
        'gym',
        'torch',
        'wandb',
        'tensorboard',
    ],
    extras_require=extras,
    tests_require=['pytest'],
    python_requires='>=3.8',
)
