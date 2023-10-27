# -*- coding: latin-1 -*-
import ez_setup
import sys
from setuptools import setup, find_packages

import growthwheat

"""
    Setup script for installation.
    
    See README.rst for installing procedure.

    :copyright: Copyright 2014-2017 INRA-ECOSYS, see AUTHORS.
    :license: CeCILL-C, see LICENSE for details.
    
"""

"""
    Information about this versioned file:
        $LastChangedBy$
        $LastChangedDate$
        $LastChangedRevision$
        $URL$
        $Id$
"""

ez_setup.use_setuptools()

if sys.version_info < (2, 7):
    print('ERROR: Growth-Wheat requires at least Python 2.7 to run.')
    sys.exit(1)

setup(
    name="Growth-Wheat",
    version=growthwheat.__version__,
    packages=find_packages(),

    install_requires=['pandas>=0.18.0'],
    include_package_data=True,

    # metadata for upload to PyPI
    author="M.Gauthier, C.Chambon, R.Barillot",
    author_email="camille.chambon@inra.fr, romain.barillot@inra.fr",
    description="Model of leaf growth for wheat",
    long_description="A mechanistic model of leaf growth for wheat that accounts for the CN status",
    license="CeCILL-C",
    keywords="functional-structural plant model, wheat, leaf growth, morphogenesis, trophic status, carbon, nitrogen, metabolism",
    url="https://sourcesup.renater.fr/projects/growth-wheat/",
    download_url="https://sourcesup.renater.fr/frs/download.php/latestfile/2184/growthwheat.zip",
)
