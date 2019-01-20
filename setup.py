from setuptools import setup, find_packages

setup(
    name="bandits",
    version="0.1.0",
    author="Byron Galbraith",
    author_email="byron.galbraith@gmail.com",
    description="Algorithms for Multi-Armed Bandits",
    url="https://github.com/bgalbraith/bandits",
    packages=[package for package in find_packages()
              if package.startswith('bandits')],
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'seaborn', 'pymc3'
    ]
)
