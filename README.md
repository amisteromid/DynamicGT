# DynamicGT
DynamicGT: a dynamic-aware geometric transformer model to predict protein binding interfaces in flexible and disordered regions

[![bioRxiv shield](https://img.shields.io/badge/bioRxiv-1709.01233-green.svg?style=flat)](https://www.biorxiv.org/content/10.1101/2025.03.04.641377v1)

![DynamicGT concept movie](states.gif)

## Table of Contents: 
- [The workflow](#The-workflow)
- [Installation](#Installation)
- [Training the model](#Training-the-model)
- [Running inference](#Running-inference)
- [License](#License)
- [Acknowledgement](#Acknowledgement)
- [Citation](#Citation)

## The workflow
The following figure illustrates the architecture and workflow of the DynamicGT pipeline. (a) DynamicGT processes the protein conformational ensemble derived from MD simulations, NMR, and AlphaFlow. The architecture employs cooperative GNNs, where the first two GNNs determine the actions of each atom, eectively constraining message passing between surface and core residues. With each layer of the architecture, the number of residues increases to emphasize closer neighbors. Subsequently, a geometric transformer updates the atomic states. The final block pools atomic information to residues and decodes this into predicted probabilities. (b) RMSF and DE are dynamic features associated with nodes, while motion V, motion s, and CP are defined for edges. RSA is calculated by averaging across multiple conformations. Additionally, displacement vectors and distances are defined to capture spatial and geometrical features of the protein structure. (c) Visualization of the broadcasting rate (red) and listening rate (blue) of surface residues, illustrating their relationship with the proximity of each residue to the interface.

![DynamicGT concept workflow](Arch.png)

## **Installation**
To set up the project, follow these steps:

```bash
git clone https://github.com/amisteromid/DynamicGT.git
conda env create -f env.yml
```
Ensure you have Git and Conda installed before proceeding.
### **Geo-Distance Calculation**
For geo-distance calculation, MSMS is a prerequisite. You can get it from [here](https://ccsb.scripps.edu/msms/downloads/).

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
4. Calculate Geo-Distances (Optional):
If not using the basic focal loss function, compute geo-distances for the geo-loss function. Refer to the script comments in loss_data folder for details.
5. Train the model:
Execute the training script with your prepared .h5 file.

## **Running inference**
To perform inference with a pre-trained model:
1. Obtain the pre-trained model from [Google Drive](https://drive.google.com/file/d/1puehNHhu6JSjH-ZZetdNaVo6ftU-Oj1x/view?usp=sharing)
2. Place your input structures in the input_structures/ folder.
3. Use the predict_with_model.ipynb Jupyter notebook to load the model and generate predictions.
4. Visualize predictions using the p_to_bfactor utility, which maps probabilities (0 to 1) to a blue-gray-red color spectrum for intuitive interpretation.

## **License**
This tool is licensed under the CC BY-NC-SA 4.0 license.

## **Acknowledgement**
We acknowledge and appreciate the contributions of the following tools, which have inspired our work and provided code that we have inspired from or adapted into our implementation:

PeSTo - Geometric Transformer<br>
GitHub: https://github.com/LBM-EPFL/PeSTo/tree/main<br>
Paper: https://doi.org/10.1038/s41467-023-37701-8<br>

CoGNN – Cooperative Graph Neural Networks<br>
GitHub: https://github.com/benfinkelshtein/CoGNN<br>
Paper: https://doi.org/10.48550/arXiv.2310.01267<br>

## **Citation**
```bibtex
@article {Mokhtari2025.03.04.641377,
	author = {Mokhtari, Omid and Grudinin, Sergei and Karami, Yasaman and Khakzad, Hamed},
	title = {DynamicGT: a dynamic-aware geometric transformer model to predict protein binding interfaces in flexible and disordered regions},
	elocation-id = {2025.03.04.641377},
	year = {2025},
	doi = {10.1101/2025.03.04.641377},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Protein-protein interactions are fundamental to cellular processes, yet existing deep learning approaches for binding site prediction often rely on static structures, limiting their performance when disordered or flexible regions are involved. To address this, we introduce a novel dynamic-aware method for predicting protein-protein binding sites by integrating conformational dynamics into a cooperative graph neural network (Co-GNN) architecture with a geometric transformer (GT). Our approach uniquely encodes dynamic features at both the node (atom) and edge (interaction) levels, and consider both bound and unbound states to enhance model generalization. The dynamic regulation of message passing between core and surface residues optimizes the identification of critical interactions for efficient information transfer. We trained our model on an extensive overall 1-ms molecular dynamics simulations dataset across multiple benchmarks as the gold standard and further extended it by adding generated conformations by AlphaFlow. Comprehensive evaluation on diverse independent datasets containing disordered, transient, and unbound structures showed that incorporating dynamic features in cooperative architecture significantly boosts prediction accuracy when flexibility matters, and requires substantially less amount of data than leading static models.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2025/03/10/2025.03.04.641377},
	eprint = {https://www.biorxiv.org/content/early/2025/03/10/2025.03.04.641377.full.pdf},
	journal = {bioRxiv}
}
```
