# %%
from scaleflow.data import DataManager, AnnDataLocation
from pathlib import Path
import anndata as ad
import h5py

print("loading data")
OUTPUT_PATH = Path("/lustre/groups/ml01/workspace/100mil/tahoe.zarr")
DATA_PATH = Path("/lustre/groups/ml01/workspace/100mil/100m_int_indices.h5ad")

with h5py.File(DATA_PATH, "r") as f:
    adata = ad.AnnData(
        obs=ad.io.read_elem(f["obs"]),
        obsm=ad.experimental.read_lazy(f["obsm"]),
        uns=ad.io.read_elem(f["uns"]),
    )

adl = AnnDataLocation()
dm = DataManager(
    dist_flag_key="control",
    src_dist_keys=["cell_line"],
    tgt_dist_keys=["drug", "dosage"],
    rep_keys={
        'cell_line': 'cell_line_embeddings',
        'drug': 'drug_embeddings',
    },
    data_location=adl.obsm['X_pca'][:,:50],
)

print("data loaded")

# %%
gd = dm.prepare_data(
    adata=adata,
)


# %%
chunk_size = 131072
shard_size = chunk_size * 8

gd.write_zarr(
    path=OUTPUT_PATH,
    chunk_size=chunk_size,
    shard_size=shard_size,
)

print("data written")