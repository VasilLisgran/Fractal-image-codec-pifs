import numpy as np
from Blocks import create_Domain_blocks, get_isometries

# Using OLS to find s and o 
def Affine_transformation(R, R_sum, domain_info):
    D = domain_info.data
    n2 = float(R.size)

    Sum_R = R_sum
    Sum_D = domain_info.sum_D
    Sum_DD = domain_info.sum_DD

    Sum_RD = np.sum(R * D)

    denominator = n2 * Sum_DD - Sum_D**2

    if abs(denominator) < 1e-9:
        s = 0.0
    else:
        s = np.clip((n2 * Sum_RD - Sum_R * Sum_D) / denominator, -0.75, 0.75)
        s = np.round((s + 0.75) / 1.5 * 30) / 30 * 1.5 - 0.75   
    o = (Sum_R - s * Sum_D) / n2
    o = np.clip(np.round(o * 63) / 63, 0.0, 1.0)                 

    diff = R - (s * D + o)
    error = np.mean(diff ** 2)
    return s, o, error

def find_best(R_block, dom):
    r = R_block.ravel().astype(np.float64)
    n2 = float(r.size)
    mr = r.mean()
    rc = r - mr
    cov = dom['D'] @ rc
    var = dom['var']

    flat = var < 1e-9
    s = np.where(flat, 0.0, cov / np.where(flat, 1.0, var))
    s = np.clip(s, -0.75, 0.75)
    s = np.round((s + 0.75) / 1.5 * 30) / 30 * 1.5 - 0.75

    o = np.clip(np.round(mr * 255) / 255, 0.0, 1.0)

    err = (rc @ rc - 2 * s * cov + s * s * var) / n2 + (mr - o) ** 2

    i = int(np.argmin(err))
    return i, float(s[i]), float(o), float(err[i])
