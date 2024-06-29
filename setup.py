from setuptools import find_packages, setup

import versioneer

setup(
    name="schedlib",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    package_dir={'': 'src'},
    packages=find_packages(where='src'),    
    install_requires=[
        "pandas",
        "pyephem",
        "jax[cpu]",
        "equinox",
        "so3g",
        "pixell",
        "dataclasses_json",
    ],
)
