# Machine Learning Fundamentals

This repository contains example notebooks for the 2025 course [Agenti Intelligenti e Machine Learning (AiTHO)](https://web.dmi.unict.it/it/corsi/l-31/agenti-intelligenti-e-machine-learning-aitho), focusing on multi-agent AI architectures.

## Tech Stack

- **Python**
- **[Marimo](https://marimo.io/)** – A modern alternative to Jupyter for interactive notebooks
- **[Pandas](https://pandas.pydata.org/)** – Python Data Analysis Library
- **[Scikit-learn](https://scikit-learn.org/)** – Tools for predictive data analysis and machine learning
- **[Tensorflow](https://www.tensorflow.org/?hl=it)** – TOpen source software library for high performance numerical computation



# Data

The dataset used in this repo are referred to insurance claims, in which each observation is labelled with fraud or not-fraud.
You can download the data in Kaggle looking for
[Insurance claims dataset](https://www.kaggle.com/buntyshah/insurance-fraud-claims-detection/data?select=insurance_claims.csv).




## Project Structure

All the example notebooks and code are located in the `notebook/` directory.

## Setup Instructions

### 1. Install Poetry

Poetry is the dependency manager used in this project. Follow the [official installation guide](https://python-poetry.org/docs/#installation) to set it up on your system.

### 2. Install Project Dependencies

```bash
poetry install
```

### 3. Launch the Notebooks
Use Marimo to edit and run the notebooks:

```bash
poetry run marimo edit
```
