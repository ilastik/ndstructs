from setuptools import setup

setup(
    name="ndstructs",
    version="0.0.1",
    author="Tomaz Vieira",
    author_email="team@ilastik.org",
    license="MIT",
    description="Short description",
    packages=["ndstructs", "ndstructs.utils", "ndstructs.datasource"],
    install_requires=["numpy", "scikit-image", "z5py", "h5py"],
)
