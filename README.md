# Anomaly-Detection-in-TESS-Planetary-Transits

The initial release consists of **TESS_Lightcurves**, a class for pre-processing and generating light curves from NASA's TESS telescope.

This class generates light curves for the TOIs available at [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/view_toi.php). It utilizes the [Lightkurve](https://docs.lightkurve.org/) package for pre-processing the light curves from the authors - SPOC, QLP, and TESS-SPOC.

## Dependencies:

It utilizes the following packages:
* [NumPy](https://numpy.org/install/)
* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [Lightkurve](https://docs.lightkurve.org/about/install.html)
* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
* [Requests](https://pypi.org/project/requests/)
* [Splinter](https://splinter.readthedocs.io/en/latest/install.html)
* [BeautifulSoup](https://pypi.org/project/beautifulsoup4/)
* [Webdriver-Manager](https://pypi.org/project/webdriver-manager/)

The above packages can be installed separately or through following commands:
> $ pip install webdriver-manager
>
> $ pip install -r requirements.txt

## Setup:

* ***download_toi.py*** will download all the available TOIs from [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/view_toi.php) as ***toi.csv***
> $ python download_toi.py
* ***read_toi.py*** will pre-process and create all the available light curves at [Lightkurve](https://docs.lightkurve.org/)
> $ python read_toi.py

This setup will create the following folders and files:
> /data
  
  * toi.pickle
  * toi_info.pickle
  * tic_info.pickle
  * tic_info_rerun.pickle

> /folded_lightcurves
  
  This folder will contain all the available TOIs with the following columns - ['phase', 'flux', 'flux_err'].
  
> /images

  This folder will contain images for all the available TOIs.

> toi.csv

> duplicate_tic_ids.csv

> unsearched_toi.pickle

> unresolved_toi.pickle
  

  

