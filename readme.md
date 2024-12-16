# A hackathon project on deconvolving SPM tip shape from scans.

We build upon the deep learning approach described in the referenced paper [1], which was originally developed to remove tip-convolution artifacts in AFM images. In this hackathon, we adapt and extend that methodology to STM datasets [2], aiming to denoise artifacts such as double-tips, drift, and line noise to recover more accurate surface topographies.








### Setup and Usage
Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
Download [STM dataset](https://doi.org/10.5281/zenodo.5799773) [2] from Zenodo using `downloader.py`:

   ```bash
   python scripts/downloader.py
   ```
Preprocess propritary Omicronscala files to numpy array:
   ```bash
   python scripts/preprocess.py
   ```

   If there is an errors with the spym package see this [PR](https://github.com/rescipy-project/spym/pull/9) .


   ### References
   1 - Bonagiri, L. K. S., Wang, Z., Zhou, S., & Zhang, Y. (2024). Precise Surface Profiling at the Nanoscale Enabled by Deep Learning. Nano Letters, 24(8), 2589-2595.
   2 - Tommaso Rodani, Elda Osmenaj, Alberto Cazzaniga, Mirco Panighel, Cristina Africh, & Stefano Cozzini. (2021). Dataset of Scanning Tunneling Microscopy (STM) images of graphene on nickel (1.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7664070

