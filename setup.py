import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

release_info = {}
python_dir = os.path.dirname(__file__)
with open(os.path.join(python_dir, "pcpm", "info.py")) as f:
    code = f.read()
    exec(code, release_info)

setuptools.setup(
    name=release_info['NAME'],
    version=release_info['__version__'],
    description=release_info['DESCRIPTION'],
    author=release_info['AUTHOR'],
    author_email=release_info['AUTHOR_EMAIL'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=release_info['URL'],
    packages=release_info['PROVIDES'],
    install_requires=release_info["REQUIRES"],
    classifiers=release_info["CLASSIFIERS"],
    python_requires=release_info['PYTHON_REQUIRES'],
    entry_points=release_info['ENTRYPOINTS'],
)
