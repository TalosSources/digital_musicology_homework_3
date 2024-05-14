# Digital Musicology (DH-401): Phrases in performance and phrases in the score

This repository contains our solution for [the third assignment](https://hackmd.io/@RFMItzZmQbaIqDdVZ0DovA/r1s0pby-R) of Digital Musicology (DH-401) course. **TODO**. We used _Schubert Impromptu Op. 90 No. 3_ in all our experiments.

We used [Aligned Scores and Performances (ASAP) dataset](https://github.com/fosfrancesco/asap-dataset) for the assignment.

## Installation

Follow this steps to reproduce our work:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:

   ```bash
   pre-commit install
   ```

3. Download dataset:

   ```bash
   mkdir data
   cd data
   git clone https://github.com/fosfrancesco/asap-dataset.git
   ```

## Project Structure

The project structure is as follows:

```bash
├── data                         # dir for all data, including raw and processed datasets
│   └── asap-dataset
├── README.md                    # this file
├── requirements.txt             # list of required packages
└── src                          # package with core implementations
│   ├── data.py
│   ├── __init__.py
│   ├── phrase_segmentation.py
│   ├── phrase_tikz.py           # segmentation to tikz converter
│   ├── ssm_helper.py            # helper function for ssm
│   └── taskC1.ipynb
└── ssm.ipynb                    # symbolic phrasing
```

## How To Use

To reproduce segmentation, run the code in the corresponding notebooks: `ssm.ipynb` and TODO.

To draw tikz picture, use `src/phrase_tikz.py`.

> [!NOTE]
> Pdf creator `phrase_tikz.py` requires `pdflatex` and `pdfcropmargins` to be installed. The later can be installed via `pip install pdfCropMargins`.

## Authors

The project was done by:

- Petr Grinberg
- Marco Bondaschi
- Ismaïl Sahbane
- Ben Erik Kriesel
