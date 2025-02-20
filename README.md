# Molecule_OPT_ML
Geometry optimization of small molecule based on spice-dataset
## Key Points
- Converts SMILES strings to 3D molecular structures using RDKit, optimizes them with a graph neural force field trained on the SPICE dataset.
- Uses SchNet from PyTorch Geometric for training, predicting energies and forces for molecular geometry optimization.
- SPICE dataset provides ground truth with energies and forces, stored in HDF5 format, accessible at [Zenodo](https://zenodo.org/record/8222043).

---

## Converting SMILES to Structures and Initial Optimization
To start, we use RDKit, a popular chemistry library, to convert SMILES strings into initial 3D molecular structures. RDKit generates coordinates in angstroms, which we convert to bohr (atomic units) to match the SPICE dataset's format, as quantum chemistry calculations typically use bohr. This initial structure, often not perfectly optimized, serves as a starting point for further refinement using our trained graph neural force field.

The process involves:
- Parsing the SMILES string with RDKit’s `Chem.MolFromSmiles`.
- Adding hydrogens with `Chem.AddHs` for completeness.
- Embedding the molecule in 3D space with `AllChem.EmbedMolecule` and optimizing with the Universal Force Field (UFF) via `AllChem.UFFOptimizeMolecule`.

We then convert the coordinates from angstroms to bohr using the factor 1.889726125 (1 angstrom ≈ 1/0.529177210903 bohr), ensuring unit consistency with the SPICE dataset.

## Training the Graph Neural Force Field
We train a SchNet model from PyTorch Geometric, a graph neural network designed for molecular systems, using the SPICE dataset as ground truth. The SPICE dataset, introduced by Eastman et al. in 2023, contains over 1.1 million conformations of drug-like molecules and peptides, with energies and forces calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory. It is available at [Zenodo](https://zenodo.org/record/8222043) in an HDF5 file named "SPICE-1.1.4.hdf5."

The training process involves:
- Loading the dataset using `h5py`, where each molecule or cluster is a top-level group containing datasets like atomic numbers, coordinates, energy, and forces.
- Creating a custom `SpiceDataset` class to map atomic numbers to indices (e.g., H=0, Li=1, etc.) and prepare data for SchNet, treating each conformation as a separate data point.
- Training SchNet with parameters like `hidden_channels=128`, `num_filters=128`, `num_interactions=3`, and `cutoff=5.0` bohr, predicting energies and computing forces as the negative gradient of energy with respect to positions.
- Minimizing a combined loss function (MSE for energy and forces) using the Adam optimizer with a learning rate of 0.001 over 100 epochs.

## Optimizing Molecule Geometry
For optimization, we start with the initial structure from the SMILES string and use the trained SchNet model to compute energies and forces. We apply the L-BFGS optimization algorithm to adjust atomic positions, minimizing the energy to find the stable, optimized geometry. This process ensures the molecule’s structure aligns with physical principles, leveraging the force field trained on high-quality SPICE data.

The optimization involves:
- Creating a `Data` object with the initial positions (in bohr) and atomic number indices.
- Defining an energy function using the SchNet model and optimizing it with L-BFGS over 10 iterations, updating positions to minimize energy.
- Converting the final optimized coordinates back to angstroms for standard representation.

## Interesting Unit Conversion Detail
It’s noteworthy that we must convert units from angstroms (RDKit output) to bohr (SPICE dataset standard), using a factor of approximately 1.889726125, highlighting the importance of unit consistency in molecular modeling for accurate predictions.

---

## Detailed Survey Note: Training a Graph Neural Force Field for Molecular Geometry Optimization

This note provides a comprehensive exploration of training a graph neural force field to optimize molecular geometry, starting from SMILES string conversion and utilizing the SPICE dataset as ground truth. The process involves several technical steps, each grounded in established computational chemistry and machine learning practices, with detailed considerations for data handling, model training, and optimization.

### Dataset Acquisition and Loading
The SPICE dataset, introduced by Eastman et al. in their 2023 publication "SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials" ([Scientific Data](https://www.nature.com/articles/s41597-022-01882-6)), is a quantum chemistry dataset designed for training machine learning potentials, particularly for drug-like small molecules interacting with proteins. It contains over 1.1 million conformations across various subsets, including small molecules, dimers, dipeptides, and solvated amino acids, with energies and forces calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory. The dataset is accessible at [Zenodo](https://zenodo.org/record/8222043), stored in a single HDF5 file named "SPICE-1.1.4.hdf5," as detailed in the paper’s "Data Records" section.

To load this dataset, we use the `h5py` library in Python, which supports HDF5 file handling. The file structure organizes data into groups, each corresponding to a molecule or cluster, with datasets for coordinates, atom types, energy, and forces. For instance, we assume keys like "coordinates" (shape \(n_{\text{atoms}}, 3\)), "atom_types" (list of atomic numbers), "energy" (scalar value), and "forces" (shape \(n_{\text{atoms}}, 3\)). This structure allows us to extract the necessary data for training, ensuring compatibility with graph neural network frameworks.

| Subset                  | Molecules/Clusters | Conformations | Atoms | Elements                              |
|-------------------------|--------------------|---------------|-------|---------------------------------------|
| Dipeptides              | 677                | 33,850        | 26–60 | H, C, N, O, S                        |
| Solvated Amino Acids    | 26                 | 1300          | 79–96 | H, C, N, O, S                        |
| DES370K Dimers          | 3490               | 345,676       | 2–34  | H, Li, C, N, O, F, Na, Mg, P, S, Cl, K, Ca, Br, I |
| DES370K Monomers        | 374                | 18,700        | 3–22  | H, C, N, O, F, P, S, Cl, Br, I       |
| PubChem                 | 28,039             | 1,398,566     | 3–50  | H, B, C, N, O, F, Si, P, S, Cl, Br, I |
| Solvated PubChem        | 1397               | 13,934        | 63–110| H, C, N, O, F, P, S, Cl, Br, I       |
| Amino Acid Ligand Pairs | 79,967             | 194,174       | 24–72 | H, C, N, O, F, P, S, Cl, Br, I       |
| Ion Pairs               | 28                 | 1426          | 2     | Li, F, Na, Cl, K, Br, I              |
| Water Clusters          | 1                  | 1000          | 90    | H, O                                 |
| Total                   | 113,999            | 2,008,628     | 2–110 | H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br, I |

This table, sourced from the GitHub repository [openmm/spice-dataset](https://github.com/openmm/spice-dataset), summarizes the dataset’s composition, highlighting its coverage of chemical space and conformations, essential for robust model training.

### Data Preparation for Graph Neural Networks
To train a graph neural network, we represent each molecule as a graph, with atoms as nodes and bonds or spatial interactions as edges. The SchNet model, implemented in PyTorch Geometric, is chosen for its effectiveness in molecular property prediction. It requires node features (atomic numbers, represented as indices), edge features (distances within a cutoff), and positional information for energy and force predictions.

We create a custom dataset class, `SpiceDataset`, extending PyTorch Geometric’s `Dataset`. This class maps atomic numbers to indices for one-hot encoding, ensuring compatibility with SchNet. For example, given the SPICE dataset’s elements (H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br, I), we identify unique atomic numbers (1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 35, 53) and create a mapping, e.g., `{1: 0, 3: 1, ..., 53: 16}`. Each data point in the dataset is converted to a `Data` object with attributes `pos` (coordinates), `z` (mapped atomic numbers), `y` (energy), and `forces` (force vectors).

### Model Training with SchNet
The SchNet model is initialized with parameters such as `num_features` (number of unique atom types, 17 in this case), `hidden_channels` (128), `num_filters` (128), `num_interactions` (3), and `cutoff` (5.0 Å, converted to bohr for consistency). Training involves minimizing a combined loss function, comprising the mean squared error (MSE) for energy predictions and force predictions. Forces are computed as the negative gradient of the predicted energy with respect to atomic positions, leveraging PyTorch’s autograd functionality.

The training loop batches data using `DataLoader` with a batch size of 32, iterating over epochs (set to 100 for demonstration). For each batch, we compute the energy prediction, derive forces via gradient computation, and update the model parameters using Adam optimizer with a learning rate of 0.001. The loss function is:

\[
\text{Loss} = \text{MSE}(\text{Energy}_{\text{pred}}, \text{Energy}_{\text{true}}) + \text{MSE}(\text{Forces}_{\text{pred}}, \text{Forces}_{\text{true}})
\]

Given the computational cost of force computation, we handle batched data by iterating over individual molecules within a batch, computing gradients per molecule, which may be inefficient but ensures accuracy for demonstration purposes.

### SMILES to Initial Structure Conversion
Starting from a SMILES string, we use RDKit to generate an initial 3D structure. The process involves:
1. Parsing the SMILES string with `Chem.MolFromSmiles`.
2. Adding hydrogens with `Chem.AddHs` for completeness.
3. Embedding the molecule in 3D space with `AllChem.EmbedMolecule`, using a random seed for reproducibility.
4. Optimizing the structure with the Universal Force Field (UFF) via `AllChem.UFFOptimizeMolecule`.

RDKit outputs coordinates in angstroms, which we convert to bohr using the factor 1.889726125 (1 angstrom ≈ 1/0.529177210903 bohr), aligning with the SPICE dataset’s units. Atomic numbers are extracted and mapped to indices using the same mapping as the training data, ensuring consistency.

### Geometry Optimization with Trained Model
For geometry optimization, we start with the initial structure and use the trained SchNet model to predict energies and compute forces. The optimization process employs the L-BFGS algorithm, implemented via PyTorch’s `optim.LBFGS`, to minimize the energy function. We define an energy function that updates the model’s input positions and computes the energy, with gradients (forces) computed via autograd. The optimization iterates for a set number of steps (10 in this example), adjusting positions to minimize energy, thus finding the stable molecular geometry.

Given SchNet’s continuous filter based on distances, we do not precompute fixed edges, allowing dynamic interaction updates during optimization. This approach, while an approximation, is standard in machine learning force fields, balancing computational efficiency with accuracy.

### Unit Considerations and Vectorization
A critical aspect is unit consistency. The SPICE dataset uses bohr for coordinates, hartrees for energies, and hartree/bohr for forces, reflecting quantum chemistry conventions. RDKit’s output in angstroms necessitates conversion, highlighting the importance of unit awareness in molecular modeling. The term "vectorize structure" likely refers to representing the structure numerically, which is inherently done by using coordinate arrays, already covered in our process.

### Challenges and Assumptions
Challenges include ensuring the SMILES string contains only atoms present in the SPICE dataset, handled by checking against the atom type mapping. We assume the HDF5 file structure matches our expectations, with specific keys for data extraction, and that the SchNet model parameters are sufficiently tuned for convergence. Future work could involve hyperparameter optimization and handling larger batches more efficiently for force computations.

This comprehensive approach ensures a robust pipeline for training a graph neural force field and optimizing molecular geometries, leveraging high-quality data and established machine learning techniques.

### Key Citations
- SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials [Scientific Data](https://www.nature.com/articles/s41597-022-01882-6)
- GitHub repository for SPICE dataset scripts and details [openmm/spice-dataset](https://github.com/openmm/spice-dataset)
- Zenodo record for SPICE dataset access [Zenodo](https://zenodo.org/record/8222043)
