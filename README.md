# DynamicGT
DynamicGT: a dynamic-aware geometric transformer model to predict protein binding interfaces in flexible and disordered regions
### Workflow explanations and figure+gif

<p align="center">
    <img src="Arch.png" width="600"/>
</p>

## **Installation**
To set up the project, follow these steps:
```bash
# Clone the repository
git clone https://github.com/amisteromid/DynamicGT.git

# Create the Conda environment using the provided env.yml file
conda env create -f env.yml
```
Ensure you have Git and Conda installed before proceeding.
## **Training the model**
Follow these steps to train the model:
1. Access the clusters of MD and AFlow data from Zenodo:
https://doi.org/10.5281/zenodo.14833854
2. Run the build_dataset.py script to generate an HDF5 (.h5) file. Use the following parameters:
   - ``--input``: path to the folder containing structures.
   - ``--output``: Path and name of the output .h5 file.
   - ``--min-sequence-length``:  Minimum sequence length to process (integer).
   - ``--num--workers``:  number of worker processes for data loading (integer).
```bash
python3 build_dataset.py --input /path/to/structures --output dataset.h5 --min-sequence-length 10 --num-workers 4
```
4. Calculate Geo-Distances (Optional)
If not using the basic focal loss function, compute geo-distances for the geo-loss function. Refer to the script comments in loss_data folder for details.
5. Train the model
Execute the training script with your prepared .h5 file.

## **Running inference**
Download the model from [Google Drive](https://drive.google.com/file/d/1puehNHhu6JSjH-ZZetdNaVo6ftU-Oj1x/view?usp=sharing)
put input structures in input_structures folder and use predict_with_model notebook to use the model. visualization can also be done using p_to_bfactor with blue to gray to red spectrum showing predicted probablity of 0 to 1.
## **Citation**
Accompanying paper
