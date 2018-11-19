from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gsdp',
    version='1.0',
    description='Global Semantic Descriptor based in Object Prototypes',
    long_description=long_description,
    author='Omar Vidal Pino',
    author_email='ovidalp@dcc.ufmg.br',
    license='Verlab',
    packages=find_packages(exclude=('config_imagenet', 'docs','dataset_tools','extractors_old')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False)
