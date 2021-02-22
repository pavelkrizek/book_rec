from pathlib import Path

from setuptools import find_packages
from setuptools import setup

setup(
    name="book_rec",
    packages=find_packages(exclude=["notebooks", "reports", "deployment", "config", "docs", "data"]),
    package_data={"book_rec": ["book_rec/package_data/*"]},
    python_requires=">=3.6",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="A short description of the project.",
    long_description=open(Path("README.md")).read(),
    long_description_content_type="text/markdown",
    author="Pavel Krizek",
    author_email="datascience@heidelbergcement.com",
    url="Url to the app repository",
)
