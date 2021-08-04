from setuptools import setup

setup(
    name="ndstructs",
    version="0.0.5dev0",
    author="Tomaz Vieira",
    author_email="team@ilastik.org",
    license="MIT",
    description="Short description",
    packages=["ndstructs", "ndstructs.utils", "ndstructs.datasource", "ndstructs.datasink", "ndstructs.caching"],
    package_data={
        'ndstructs': ['py.typed']
    },
    python_requires=">=3.7",
    install_requires=["numpy", "scikit-image", "typing_extensions"],
    extras_require={"dev": ["pytest<5.4"]},
)
