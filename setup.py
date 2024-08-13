from setuptools import setup

setup(
    name="ndstructs",
    version="0.0.6",
    author="Tomaz Vieira",
    author_email="team@ilastik.org",
    license="MIT",
    description="Short description",
    packages=["ndstructs", "ndstructs.utils", "ndstructs.datasource", "ndstructs.datasink", "ndstructs.caching"],
    python_requires=">=3.8",
    install_requires=["numpy", "scikit-image", "h5py", "fs", "typing_extensions", "imageio"],
    extras_require={"dev": ["pytest"]},
)
