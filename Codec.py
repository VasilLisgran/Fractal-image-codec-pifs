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

def compress_quadtree(img, thresh=0.00003, B_max=16, B_min=4):
    domain_cache = prepare_domain_cache(img, sizes=(16, 8, 4))
    transforms = []

    def process(y, x, size):
        dom = domain_cache[size]
        i, s, o, best_error = find_best(img[y:y+size, x:x+size], dom)

        if best_error <= thresh or size <= B_min:
            transforms.append({'y': y, 'x': x, 'size': size,
                               'j': int(dom['j'][i]), 'k': int(dom['k'][i]),
                               's': s, 'o': o})
        else:
            half = size // 2
            process(y, x, half)
            process(y, x + half, half)
            process(y + half, x, half)
            process(y + half, x + half, half)

    H, W = img.shape
    for y in range(0, H, B_max):
        for x in range(0, W, B_max):
            process(y, x, B_max)
    return transforms

def decompress_quadtree(transforms, img_shape=(256, 256), num_iterations=20, default_val=0.5):
    H, W = img_shape
    img = np.full((H, W), default_val, dtype=np.float32)
    sizes_used = {t['size'] for t in transforms}
    for _ in range(num_iterations):
        doms = {sz: build_domains(img, sz) for sz in sizes_used}
        new_img = np.zeros((H, W), dtype=np.float32)
        for t in transforms:
            y, x, size = t['y'], t['x'], t['size']
            D = doms[size][t['j']]
            D_iso = get_isometries(D)[t['k']]
            new_img[y:y + size, x:x + size] = t['s'] * (D_iso - D.mean()) + t['o']
        img = np.clip(new_img, 0.0, 1.0)
    return img
