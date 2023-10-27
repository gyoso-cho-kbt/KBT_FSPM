# -*- coding: latin-1 -*-
import ez_setup
import sys
from setuptools import setup, find_packages

import senescwheat

"""
Notes:

- use setup.py develop when tracking in-development code
- when removing modules or data files from the project, run setup.py clean --all and delete any obsolete .pyc or .pyo.

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
    print('ERROR: Senesc-Wheat requires at least Python 2.7 to run.')
    sys.exit(1)

setup(
    name="Senesc-Wheat",
    version=senescwheat.__version__,
    packages=find_packages(),

    install_requires=['pandas>=0.18.0'],
    include_package_data=True,

    # metadata for upload to PyPI
    author="M.Gauthier, C.Chambon, R.Barillot",
    author_email="camille.chambon@inra.fr, romain.barillot@inra.fr",
    description="Model of senescence adapted to wheat",
    long_description="Model of senescence adapted to wheat",
    license="CeCILL-C",
    keywords="senescence, tissue death, trophic regulation, green area, leaf, roots, nitrogen, remobilisation",
    url="https://sourcesup.renater.fr/projects/senesc-wheat/",
    download_url="https://sourcesup.renater.fr/frs/download.php/latestfile/1923/senesc-wheat_BreedWheat.zip",
)
