# High-Throughput Cell Colony & Caspase Quantification

This repository contains the image processing pipeline used in Cell Death Project. It utilizes FastSAM and Cellpose to perform structural boundary detection and high-density nuclear quantification on micropatterned cellular substrates. 

It does quantification of several metrics by averaging similar colonies, quantifying radial metrics (object distance from centroid, movement towards center) and does statistics on resulting measurements. 

## Installation
```bash
git clone [https://github.com/YourUsername/Cell_Colony_Pipeline.git](https://github.com/YourUsername/Cell_Colony_Pipeline.git)
cd CellDeathProject
pip install -r requirements.txt