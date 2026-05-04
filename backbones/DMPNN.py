import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import Mol, MolFromSmiles


class DMPNN_Fingerprint:
    def __init__(self, mol_dim: int, device: str | torch.device | None = None):
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        mp_path = "./DMPNN_pretrained.pt"

        dmpnn_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**dmpnn_mp["hyper_parameters"])
        mp.load_state_dict(dmpnn_mp["state_dict"])
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),  # not actually used
        )
        self.model.eval()

        self.DimReduce = torch.nn.Linear(2048, mol_dim)
        if device is not None:
            self.model.to(device=device)
            self.DimReduce.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        bmg = BatchMolGraph(
            [
                self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m)   # uses RDKit's SMILES featurizer
                for m in molecules
            ]
        )
        bmg.to(device=self.model.device)
        emb = self.model.fingerprint(bmg)
        return self.DimReduce(emb)
