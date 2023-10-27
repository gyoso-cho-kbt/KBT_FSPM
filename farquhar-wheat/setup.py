# -*- coding: latin-1 -*-

import ez_setup
import sys
from setuptools import setup, find_packages

import farquharwheat

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
    print('ERROR: Farquhar-Wheat requires at least Python 2.7 to run.')
    sys.exit(1)

setup(
    name="Farquhar-Wheat",
    version=farquharwheat.__version__,
    packages=find_packages(),

    install_requires=['pandas>=0.18.0', 'numpy'],
    include_package_data=True,

    # metadata for upload to PyPI
    author="C.Chambon, R.Barillot",
    author_email="camille.chambon@inra.fr, romain.barillot@inra.fr",
    description="Farqhuar model of wheat photosynthesis",
    long_description="Farqhuar model of wheat photosynthesis",
    license="CeCILL-C",
    keywords="photosynthesis, Farquhar, stomatal conductance, light, CO2, temperature, transpiration",  # TODO
    url="https://sourcesup.renater.fr/projects/farquhar-wheat/",
    download_url="https://sourcesup.renater.fr/frs/download.php/latestfile/1737/farquhar-wheat_BreedWheat.zip",
)
