import numpy as np
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import SchNet
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.optim as optim

# Conversion factor: angstroms to bohr
ANG2BOHR = 1.889726125

# Atomic number to index mapping based on SPICE dataset elements (H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br, I)
ATOM_MAP = {1: 0, 3: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 11: 7, 12: 8, 
            14: 9, 15: 10, 16: 11, 17: 12, 19: 13, 20: 14, 35: 15, 53: 16}

# Custom dataset class for SPICE
class SpiceDataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        super(SpiceDataset, self).__init__()
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.data_list = []
        self._load_data()

    def _load_data(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            for group_name in list(f.keys())[:100]:  # Limit to 100 groups for demo (remove for full training)
                group = f[group_name]
                # Assuming dataset structure: coordinates, atom_types, energy, forces
                pos = torch.tensor(group['coordinates'][:], dtype=torch.float32)  # in bohr
                atom_types = group['atom_types'][:]
                z = torch.tensor([ATOM_MAP[at] for at in atom_types], dtype=torch.long)
                energy = torch.tensor(group['energy'][:], dtype=torch.float32)  # in hartree
                forces = torch.tensor(group['forces'][:], dtype=torch.float32)  # in hartree/bohr
                data = Data(pos=pos, z=z, y=energy, forces=forces)
                self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        return data

# Function to convert SMILES to initial 3D structure
def smiles_to_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    mol = Chem.AddHs(mol)  # Add hydrogens
    AllChem.EmbedMolecule(mol, randomSeed=42)  # Initial 3D embedding
    if mol.GetNumConformers() == 0:
        raise ValueError("Failed to embed molecule in 3D")
    AllChem.UFFOptimizeMolecule(mol)  # Optimize with UFF
    conf = mol.GetConformer()
    pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])  # in angstroms
    pos = torch.tensor(pos, dtype=torch.float32) * ANG2BOHR  # Convert to bohr
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    # Check for unsupported atoms
    unsupported = set(atom_types) - set(ATOM_MAP.keys())
    if unsupported:
        raise ValueError(f"Unsupported atom types in SMILES: {unsupported}")
    z = torch.tensor([ATOM_MAP[at] for at in atom_types], dtype=torch.long)
    return pos, z

# Training function
def train_model(model, train_loader, device, epochs=10):  # Reduced epochs for demo
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # Predict energy (SchNet expects batch argument for batched data)
            energy_pred = model(batch.z, batch.pos, batch=batch)
            # Compute forces as negative gradient of energy w.r.t. positions
            forces_pred = -torch.autograd.grad(energy_pred.sum(), batch.pos, create_graph=True)[0]
            # Loss: MSE for energy and forces
            energy_loss = torch.nn.functional.mse_loss(energy_pred, batch.y)
            forces_loss = torch.nn.functional.mse_loss(forces_pred, batch.forces)
            loss = energy_loss + forces_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")
    return model

# Geometry optimization function
def optimize_geometry(model, pos, z, device, steps=10):
    model.eval()
    pos = pos.clone().detach().requires_grad_(True).to(device)
    z = z.to(device)
    optimizer = optim.LBFGS([pos], lr=0.1, max_iter=20)

    def closure():
        optimizer.zero_grad()
        data = Data(pos=pos, z=z).to(device)
        energy = model(data.z, data.pos)  # Single molecule, no batch
        energy.backward()
        return energy

    for step in range(steps):
        optimizer.step(closure)
        print(f"Optimization Step {step+1}/{steps}, Energy: {closure().item():.6f}")
    return pos.detach() / ANG2BOHR  # Convert back to angstroms

# Prediction function: Input SMILES, output optimized structure
def predict_optimized_structure(smiles, model, device):
    try:
        # Convert SMILES to initial structure
        pos, z = smiles_to_structure(smiles)
        # Optimize geometry using trained model
        optimized_pos = optimize_geometry(model, pos, z, device)
        return optimized_pos.cpu().numpy()  # Return as numpy array in angstroms
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SPICE dataset (adjust path as needed)
    hdf5_path = "path/to/SPICE-1.1.4.hdf5"  # Replace with actual path
    print(f"Loading dataset from {hdf5_path}")
    dataset = SpiceDataset(hdf5_path)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} conformations")

    # Initialize SchNet model
    model = SchNet(
        num_features=len(ATOM_MAP),  # 17 unique atom types
        hidden_channels=128,
        num_filters=128,
        num_interactions=3,
        cutoff=5.0,  # in bohr
        num_gaussians=50
    ).to(device)

    # Train the model
    print("Starting training...")
    model = train_model(model, train_loader, device)

    # Example prediction
    smiles = "CCO"  # Ethanol
    print(f"\nPredicting optimized structure for SMILES: {smiles}")
    optimized_coords = predict_optimized_structure(smiles, model, device)
    if optimized_coords is not None:
        print("Optimized coordinates (angstroms):")
        print(optimized_coords)

if __name__ == "__main__":
    main()