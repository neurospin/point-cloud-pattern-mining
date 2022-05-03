# -*- coding: utf-8 -*-

version_major = 0
version_minor = 1
version_micro = 0
version_extra = ''

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (version_major,
                              version_minor,
                              version_micro,
                              version_extra)
CLASSIFIERS = [
    "Programming Language :: Python",
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    "Operating System :: OS Independent",
]


description = "Tools for the calculation of moving averages of 3D objects"

# versions for dependencies
SPHINX_MIN_VERSION = '1.0'

# Main setup parameters
NAME = 'point_cloud_pattern_mining'
PROJECT = 'point-cloud-pattern-mining'
ORGANISATION = "neurospin"
MAINTAINER = "nobody"
MAINTAINER_EMAIL = "support@neurospin.info"
DESCRIPTION = description
URL = "https://github.com/neurospin/point-cloud-pattern-mining"
DOWNLOAD_URL = "https://github.com/neurospin/point-cloud-pattern-mining"
LICENSE = "CeCILL-B"
AUTHOR = 'Marco Pascucci - NeuroSpin (CEA)'
AUTHOR_EMAIL = 'marpas.paris@gmail.com'
PLATFORMS = "OS Independent"
PROVIDES = ['pcpm', 'pcpm.distance', 'pcpm.isomap',
            'pcpm.cli', 'pcpm.average', 'pcpm.recipes']
REQUIRES = ['scipy>=0.22', "pandas", "numpy", "tqdm", "sklearn", "matplotlib"]
EXTRA_REQUIRES = {
    "doc": ["sphinx>=" + SPHINX_MIN_VERSION],
}
# from python 3.6, dictionary order is preserved during execution
PYTHON_REQUIRES = '>=3.6'
ENTRYPOINTS = {
    'console_scripts': [
        'pcpm_dist=pcpm.dist.icp:main'
    ],
}

brainvisa_build_model = 'pure_python'
