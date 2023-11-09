from setuptools import find_packages, setup

setup(
    name="autointerp",
    version=0.1,
    description="",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={},
    scripts=[],
    install_requires=["transformer-lens"],
    entry_points={
        "console_scripts": [],
    },
    classifiers=[],
    tests_require=["pytest"],
    setup_requires=["pytest-runner"],
    keywords="",
)
