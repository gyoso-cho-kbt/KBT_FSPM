# -*- coding: latin-1 -*-

import ez_setup
import sys
from setuptools import setup, find_packages

import respiwheat

"""

    setup
    ~~~~~
    
    Setup script for installation.
    
    See README.rst for installing procedure.

    :copyright: Copyright 2014-2017 INRA-ECOSYS, see AUTHORS.
    :license: CeCILL-C, see LICENSE for details.
    
    **Acknowledgments**: The research leading these results has received funding through the 
    Investment for the Future programme managed by the Research National Agency 
    (BreedWheat project ANR-10-BTBR-03).
    
    .. seealso:: Barillot et al. 2016.
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
    print('ERROR: Respi-Wheat requires at least Python 2.7 to run.')
    sys.exit(1)

setup(
    name="Respi-Wheat",
    version=respiwheat.__version__,
    packages=find_packages(),

    include_package_data=True,

    # metadata for upload to PyPI
    author="C.Chambon, R.Barillot",
    author_email="camille.chambon@inra.fr, romain.barillot@inra.fr",
    description="Model of respiration adapted to wheat ",
    long_description="Model of respiration adapted to the main physiological processes of wheat. Adapted from Cannell et Thornley (2000).",
    license="CeCILL-C",
    keywords="respiration model",
    url="https://sourcesup.renater.fr/projects/respi-wheat/",
    download_url="https://sourcesup.renater.fr/frs/download.php/latestzip/2087/Respi-Wheat-Stable-latest.zip",
)
