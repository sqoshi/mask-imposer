from setuptools import find_packages, setup

setup(
    name="Mask Imposer",
    version="2.4.0",
    description="Tool to overlay fake face masks.",
    url="https://github.com/sqoshi/mask-imposer",
    author="Piotr Popis",
    author_email="piotrpopis@icloud.com",
    license="MIT",
    py_modules=["run"],
    packages=["mask_imposer"] + list(find_packages()),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "mim = run:main",
        ]
    },
    include_package_data=True,
    install_requires=[
        "wheel>=0.36.2",
        "termcolor>=1.1.0",
        "numpy>=1.21.1",
        "opencv-python>=4.5.3.5",
        "coloredlogs>=15.0.1",
        "landmark-predictor @ git+https://github.com/sqoshi/landmark-predictor.git",
    ],
)
