# Stellar Classification: stars, quasars and galaxies

##### [Giovanni Milone](https://github.com/Archstetics), [Christian Gennarelli](https://github.com/ilGennaa) <br> AI Lab, Prof. Daniele Pannone <br> La Sapienza University of Rome <br> Applied CS and AI, 2023.

The main goal of the project was to find build a proper **multi-class classifier** for predicting stars, quasars and galaxies by their spectral characteristics and other relevant features, such as the position specified by the alpha and delta angles at J2000 epoch. <br>
The model chosen for this purpose was **Random Forest**, and we compared its results to the ones obtained by constructing a single **Decision Tree** under some particular circumstances.

#### Codebase Structure
All modules in the codebase were built by our team.

    ├── dataset                       # Dataset folder.
      ├── star_classification.csv
    ├── main.ipynb                    # Main Jupyter notebook: here's the implementation of the modules. All tests can be run here.
    ├── metrics_tools.py              # Module made to collect all the functions for making computations over data and plots.
    ├── models_testing.py             # Module containing functions for Grid Search.
    ├── scaling.py                    # Module with a single function for scaling and Hybrid Sampling.
    └── README.md
#### Dataset
The dataset used was found on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).  It is provided by **Sloan Digital Sky Survey**, in particular we used **SDSS17**.