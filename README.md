# DynamicGT
DynamicGT: a dynamic-aware geometric transformer model to predict protein binding interfaces in flexible and disordered regions
### Workflow

## **Installation**
1. Clone github repo
2. yml file to install the environment
```bash
git clone <>
conda env create -f env.yml
```
## **Training the model**
1. Zenodo link to clusters of MD and Aflow
2. How to run build_dataset.py with arguments and make h5 file
   The parameters of the script are:
   - ``--input``: path to the folder containing structures.
   - ``--output``: path/name of the h5 file to be saved.
   - ``--min-sequence-length``:  Minimum sequence length to process.
   - ``--num--workers``:  number of worker processes for data loading.
4. How to calculate the geo-dists for geoloss function
5. Train the model

## **Running inference**

## **Citation**
Accompanying paper
