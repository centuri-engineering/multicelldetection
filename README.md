# Confocal image processing and analysis
Python scripts for confocal image processing and analysis

----------------------------------------

## Institute: Biology of Ciliated Epithelia (IBDM)

# Submission date: 27/08/2024
## Project Title
--- | --- | Automatic cell type classification for ciliated epithelium

## Project summary

Ciliated epithelia are entrusted with a variety of roles in humans and are thus also implicated in quite a few disorders/diseases. In order to improve our understanding of its development and normal activities, we are studying the epithelium on the skin of Xenopus laevis tadpoles, which allows for easier studies/manipulations. Until now, 4 cell types (goblet cells, MCCs, ISCs and SSCs) – excluding basal cells in the inner layer – have been identified in the surface epithelium. To start with, we would like to get an idea about the proportions of the different cells under different conditions, so that we can understand the importance of the individual cells and their interactions for the normal function of the epithelium. 

## Requested task

We would be working with fluorescence images (confocal microscopy) of different parts of the embryo under different conditions. The images would have immunostaining against phalloidin or ZO-1 (marks the cell boundaries), AcTub (marks the MCCs) and PNA (marks the goblet cells and SSCs). We would like to quantify the proportions of the 4 different cell types (so, 4 classes + maybe 1 ‘ambiguous’ class for the cells that can’t be identified) based on their shape/size/surface texture in the images. Automation of this task would allow us to save time and increase the sample size.

## Installation

(Recommend) Create a conda environment:

    conda create -n multicelldetection_env python=3.11
    conda activate multicelldetection_env

Install PyTorch with GPU (for using cellpose)

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install Cellpose

    python -m pip install cellpose[gui]
    
Install the codes

    pip install git+https://github.com/ledvic/multicelldetection.git

## Updating

conda activate multicelldetection_env
pip install -U git+https://github.com/ledvic/multicelldetection.git
