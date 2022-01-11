#! /usr/bin/env python
"""A template."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('torch_explain', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'torch_explain'
DESCRIPTION = 'PyTorch Explain: Logic Explained Networks in Python.'
with codecs.open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'P. Barbiero'
MAINTAINER_EMAIL = 'barbiero@tutanota.com'
URL = 'https://github.com/pietrobarbiero/pytorch_explain'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/pietrobarbiero/pytorch_explain'
VERSION = __version__
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'scikit-learn',
    'pandas',
    'torch>=1.9',
    'sympy',
    'pytorch_lightning',
]
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved',
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/x-rst',
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
