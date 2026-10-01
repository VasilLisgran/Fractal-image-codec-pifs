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

def compress(img):
    R_blocks = convert_to_block(img)
    D_blocks = create_Domain_blocks(img)

    transforms = []
    num_R = len(R_blocks)
    num_D = len(D_blocks)

    for i in range(num_R):
        R = R_blocks[i]
        best_error = float('inf')
        best_transform = None

        for j in range(num_D):
            D = D_blocks[j]
            D_isometries = get_isometries(D)

            for k in range(8):
                D_iso = D_isometries[k]
                s, o, error = Affine_transformation(R, D_iso)

                if error < best_error:
                    best_error = error
                    best_transform = (j, k, s, o)

        transforms.append(best_transform)

        if (i + 1) % 512 == 0:
            print(f"Обработано {i + 1}/{num_R} блоков...")

    return np.array(transforms, dtype=object)
