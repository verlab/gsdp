from setuptools import setup, find_packages
# gsdp
from gsdp import __version__
from prepare_data import download_gsdp_data,models_resources

# setup -----------------------------------------------------------------------------
with open("README.md", "r") as fh:
    long_description = fh.read()

#download_gsdp_data()

# pip install setup
setup(
    name='gsdp',
    version= str(__version__),
    description='Global Semantic Descriptor based in Object Prototypes',
    long_description=long_description,
    author='Omar Vidal Pino',
    author_email='ovidalp@dcc.ufmg.br',
    keywords="semantic description prototypes objects",
    url="https://www.verlab.dcc.ufmg.br/global-semantic-description/wacv2019/",   # project home page
    license='GPLv3',
    packages=find_packages(exclude=('docs')),
    include_package_data=True,
    package_data={
                  # data package
                 'data': ['ImageNet/*.*'],
                  # models package
                 'models': models_resources(), #['MNIST/data/*.h5', 'VGG16/data/*.h5','ResNet50/data/*.h5'],
                  # test package
                 'test': ['imgs/*.jpg','imgs/*.png', '*.py'],
                 },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False)

