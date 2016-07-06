from setuptools import setup, find_packages

setup(
    name='ac_pysmac',
    packages=find_packages(),
    install_requires=['docutils>=0.3', 'setuptools', 'numpy'],
    author="Nacim Belkhir (python interface). Frank Hutter, Holger Hoos, Kevin Leyton-Brown, Kevin Murphy and Steve Ramage (SMAC)",
    description="this package is an interface to the algorithm configuration tool SMAC ",
    include_package_data=True,
    keywords="hyperparameter parameter optimization hyperopt bayesian smac global",
    license="SMAC is free for academic & non-commercial usage. Please contact Frank Hutter(fh@informatik.uni-freiburg.de) to discuss obtaining a license for commercial purposes.",
    url="http://www.cs.ubc.ca/labs/beta/Projects/SMAC/",
    extras_require={
        'analyzer': ['matplotlib'],
    },
)
