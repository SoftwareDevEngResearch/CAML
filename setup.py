"""Setup script to build module. Templated from pymars"""
#!/usr/bin/env python3

import setuptools
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

install_requires = [
    'numpy',
    'pyyaml'
]

tests_require = [
    'pytest'
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setuptools.setup(
    name="CAML",
    version='0.0.1',
    author="Morgan Mayer",
    author_email="mayermo@oregonstate.edu",
    description=(
        "Performs data cleaning and feature transformations for machine learning comparison studies. Uses the AutoML software TPOT, based in Scikit-Learn, for machine learning."
        ),
    long_description=readme,
    long_description_content_type="text/markdown",
    license='MIT',
    keywords=['machine learning'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    python_requires='>=3',
    zip_safe=False,
    )