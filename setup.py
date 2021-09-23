from setuptools import find_packages, setup


def get_version() -> str:
    with open("fastface/version.py", "r") as foo:
        version = foo.read().split("=")[-1].replace("'", "").replace('"', '').strip()
    return version


__author__ = {"name": "Ömer BORHAN", "email": "borhano.f.42@gmail.com"}

# load long description
with open("README.md", "r") as foo:
    long_description = foo.read()

# load requirements
with open("requirements.txt", "r") as foo:
    requirements = foo.read().split("\n")

docs_require = [
    "sphinxemoji",
    "sphinx_rtd_theme",
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinxcontrib-napoleon",
]

test_require = [
    "pytest",
    "pytest-pylint",
    "pytest-cov",
]

dev_require = (
    [
        "isort",
        "black",
        "flake8",
        "tox",
        "tox-conda",
    ]
    + test_require
    + docs_require
)

extras_require = {
    "test": test_require,
    "docs": docs_require,
    "dev": dev_require,
    "all": dev_require,
}

setup(
    # package name `pip install fastface`
    name="fastface",
    # package version `major.minor.patch`
    version=get_version(),
    # small description
    description="A face detection framework for edge devices using pytorch lightning",
    # long description
    long_description=long_description,
    # content type of long description
    long_description_content_type="text/markdown",
    # source code url for this package
    url="https://github.com/borhanMorphy/light-face-detection",
    # project urls
    project_urls={
        "Documentation": "https://fastface.readthedocs.io/en/latest/",
    },
    # author of the repository
    author=__author__["name"],
    # author's email adress
    author_email=__author__["email"],
    # package license
    license="MIT",
    # package root directory
    packages=find_packages(),
    # requirements
    install_requires=requirements,
    # extra requirements
    extras_require=extras_require,
    include_package_data=True,
    # keywords that resemble this package
    keywords=["pytorch_lightning", "face detection", "edge AI", "LFFD"],
    zip_safe=False,
    # classifiers for the package
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
