from setuptools import find_packages, setup

setup(
    name="Mask Imposer",
    version="1.0.0",
    description="Tool to overlay fake face masks.",
    url="https://github.com/sqoshi/mask-imposer",
    author="Piotr Popis",
    author_email="piotrpopis@icloud.com",
    license="MIT",
    py_modules=["run"],
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "mim = run:main",
        ]
    },
    install_requires=[
        "wheel==0.36.2",
        "termcolor==1.1.0",
        "dlib==19.22.0",
        "progressbar==2.5",
        "opencv-python>=4.5.3.5"
    ],
)
