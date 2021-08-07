import pandas as pd
import numpy as np
from scipy import linalg

if __name__ == '__main__':

    def compute_centroid_index(filename: str):
        # Assume centroid is (0,0,0) as SHOT descriptor computation centred PCs
        dm_esf = pd.read_csv(filename, header=None)
        centroid = dm_esf.median(axis=0)
        dm_esf -= centroid
        # print(dm_esf.median(axis=0))
        assert dm_esf.shape[1] == 3
        l2 = dm_esf.apply(lambda x: np.sqrt(x.dot(x)), axis=1)
        idx=l2.idxmin()
        return idx

    def test_compute_centroid():
        assert compute_centroid_index("matrix.txt") == 3
    test_compute_centroid()

    ESF_PATH = "../descriptors/tactile_split5/filtered/spam0_esf.csv"
    SHOT_PATH = "../descriptors/tactile_split5/filtered/spam0_shot.csv"
    PC_PATH = "../datasets/tactile_split3/filtered/baseball0.csv"

    # Load ESF descriptor
    dm_esf = pd.read_csv(ESF_PATH, header=None)
    dv_esf = pd.DataFrame(dm_esf.stack())
    assert dv_esf.shape == (640, 1)

    # Load SHOT descriptor
    dm_shot = pd.read_csv(SHOT_PATH, header=None)
    idx = compute_centroid_index(PC_PATH) # SHOT descriptor closest to the centroid
    dv_closest_shot=dm_shot.iloc[idx, :]
    assert dv_closest_shot.shape == (353,)

    zeros = pd.DataFrame(np.zeros((640-353)))
    dv_closest_shot_padded = pd.concat([dv_closest_shot, zeros])
    assert dv_closest_shot_padded.shape == (640, 1)

    # Reset indices
    dv_esf.reset_index(drop=True, inplace=True)
    dv_closest_shot_padded.reset_index(drop=True, inplace=True)

    # Create D
    D = pd.concat([dv_esf, dv_closest_shot_padded], axis=1, join='inner')
    assert D.shape == (640, 2)

    # Create Dhat by centring D
    mean = D.mean(axis=0)
    Dhat = D - mean
    assert Dhat.shape == (640, 2)

    arr = Dhat.to_numpy(na_value=0)


    # Perform SVD
    u, s, vh = linalg.svd(arr, full_matrices=True, compute_uv=True) # unitary array u, singular value vector s,
    # print(u, s, vh)

    # Natales descriptor
    cmsf_descriptor = u[:, 0] * s[0]
    print(cmsf_descriptor.shape)




