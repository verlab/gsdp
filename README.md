# Global Semantic Descriptor based on Prototypes
[![Version](https://img.shields.io/badge/version-1.0-brightgreen.svg)](https://www.verlab.dcc.ufmg.br/global-semantic-description)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

# Project #

This project contains the code and data used to generate the results reported in the paper [Prototypicality effects in global semantic description of objects](https://www.verlab.dcc.ufmg.br/global-semantic-description/wacv2019/) on the **IEEE Winter Conference on Applications of Computer Vision (WACV) 2019**. It implements a global semantic description of object using semantic prototypes of objects categories.

For more information and visual results, please access the [project page](https://www.verlab.dcc.ufmg.br/global-semantic-description/).

## Contact ##

### Authors ###

* Omar Vidal Pino - PhD student - UFMG - ovidalp@dcc.ufmg.br
* Erickson Rangel do Nascimento - Advisor - UFMG - erickson@dcc.ufmg.br
* Mario Fernando Montenegro Campos - Advisor - UFMG - mario@dcc.ufmg.br

### Institution ###

Federal University of Minas Gerais (UFMG)  
Computer Science Department  
Belo Horizonte - Minas Gerais -Brazil 

### Laboratory ###

![VeRLab](https://www.dcc.ufmg.br/dcc/sites/default/files/public/verlab-logo.png)

**VeRLab:** Laboratory of Computer Vision and Robotics   
https://www.verlab.dcc.ufmg.br

## Citation ##

If you are using GSDP descriptor for academic purposes, please cite:

     Prototypicality effects in global semantic description of objects
     Omar Vidal Pino, Erickson R. Nascimento, Mario F. M. Campos
     IEEE Winter Conference on Applications of Computer Vision (WACV), 2019
     
### Bibtex entry ###

>@InProceedings{vidal2019wacv,  
>title = {Prototypicality effects in global semantic description of objects},  
booktitle = {2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},  
>author = {Omar Vidal Pino and Erickson R. Nascimento and Mario F. M. Campos},  
>Year = {2019},  
>Address = {Hawaii, USA},  
>month = {January},  
>intype = {to appear in},  
>pages = {},  
>volume = {},  
>number = {},  
>doi = {},  
>ISBN = {}  
>}

     

## GSDP package ##
![Version 3.0](https://img.shields.io/pypi/pyversions/Django.svg)

### Dependencies ###

* Keras 2.0  _(Tested with 2.0.4)_  
* Tensorflow 1.1 _(Tested with 1.1.0)_
* Matplotlib 2.0 _(Tested with 2.0.2)_  
* H5py 2.7 _(Tested with 2.7.0)_ 
* Pandas 0.20 _(Tested with 0.20.3)_ 

### Installation ###

Installation for Python 3 environment (Python3 and pip3 active):

    git clone git://github.com/verlab/gsdp
    cd gsdp/
    pip install -r requirements.txt 
    python prepare_data.py
    pip install .
   
Installation without Python 3 environment:

    git clone git://github.com/verlab/gsdp
    cd gsdp/
    pip3 install -r requirements.txt 
    python3 prepare_data.py
    pip3 install .
    
Read more at [GSDP documentation](https://verlab.github.io/gsdp/).
