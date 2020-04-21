from setuptools import setup

setup(
    name="ndstructs",
    version="0.0.2dev0",
    author="Tomaz Vieira",
    author_email="team@ilastik.org",
    license="MIT",
    description="Short description",
    packages=["ndstructs", "ndstructs.utils", "ndstructs.datasource", "ndstructs.datasink", "ndstructs.caching"],
    install_requires=["numpy", "scikit-image", "h5py", "fs", "typing_extensions"],
)
