import numpy as np
from Blocks import convert_to_block, create_Domain_blocks, get_isometries
from Colours import rgb_to_ycbcr, ycbcr_to_rgb, subsample_2x, upsample_2x


class DomainInfo:
    __slots__ = ('data', 'sum_D', 'sum_DD')
    def __init__(self, data):
        self.data = data
        self.sum_D = float(np.sum(data))
        self.sum_DD = float(np.sum(data ** 2))

# Using OLS to find s and o 
'''
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
        s = np.round((s + 0.75) / 1.5 * 30) / 30 * 1.5 - 0.75   # как в файле
    o = (Sum_R - s * Sum_D) / n2
    o = np.clip(np.round(o * 63) / 63, 0.0, 1.0)                 # как в файле

    diff = R - (s * D + o)
    error = np.mean(diff ** 2)
    return s, o, error
'''

def resize_block(block, target_size):
    current_size = block.shape[0]
    if current_size == target_size:
        return block
    elif current_size < target_size:
        factor = target_size // current_size
        return np.repeat(np.repeat(block, factor, axis=0), factor, axis=1)
    else:
        factor = current_size // target_size
        return block[::factor, ::factor]

def pool2(img):
    h, w = img.shape
    return img[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

def build_domains(img, s, max_count=1024):
    # домен 2s×2s, усреднённый до s×s; не больше 1024 штук (j занимает 10 бит)
    P = pool2(img)
    step = max(s // 2, 1)
    while ((P.shape[0] - s) // step + 1) * ((P.shape[1] - s) // step + 1) > max_count:
        step *= 2
    return np.array([P[y:y + s, x:x + s]
                     for y in range(0, P.shape[0] - s + 1, step)
                     for x in range(0, P.shape[1] - s + 1, step)])

def prepare_domain_cache(img, sizes=(16, 8, 4)):
    cache = {}
    for s in sizes:
        rows, ids = [], []
        for j, D in enumerate(build_domains(img, s)):
            for k, D_iso in enumerate(get_isometries(D)):
                rows.append(D_iso.ravel())
                ids.append((j, k))
        ids = np.array(ids)
        M = np.array(rows, dtype=np.float64)
        Mc = M - M.mean(axis=1, keepdims=True)          # домены без среднего
        cache[s] = {'D': Mc,
                    'var': np.einsum('ij,ij->i', Mc, Mc),
                    'j': ids[:, 0], 'k': ids[:, 1]}
    return cache

def scale_transforms(transforms, scale):
    """Масштабирует координаты и размеры блоков для декодирования в другом разрешении."""
    if scale == 1.0:
        return transforms
    return [
        {
            'y': int(round(t['y'] * scale)),
            'x': int(round(t['x'] * scale)),
            'size': int(round(t['size'] * scale)),
            'j': t['j'],
            'k': t['k'],
            's': t['s'],
            'o': t['o']
        }
        for t in transforms
    ]

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
'''
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
'''
'''
def decompress(transforms, img_shape=(256, 256), num_iterations=10):
    H, W = img_shape
    R_size = 4
    blocks_per_h = H // R_size
    blocks_per_w = W // R_size
    num_blocks = blocks_per_h * blocks_per_w
    
    img_decompressed = np.full((H, W), 0.5, dtype=np.float32)

    for iteration in range(num_iterations):
        D_blocks = create_Domain_blocks(img_decompressed)
        new_R_blocks = np.zeros((num_blocks, R_size, R_size), dtype=np.float32)

        for i in range(len(transforms)):
            j, k, s, o = transforms[i]
            D = D_blocks[int(j)]
            D_iso = get_isometries(D)[int(k)]
            new_R_blocks[i] = s * D_iso + o

        new_R_blocks = new_R_blocks.reshape(blocks_per_h, blocks_per_w, R_size, R_size)
        new_R_blocks = new_R_blocks.transpose(0, 2, 1, 3)
        img_decompressed = new_R_blocks.reshape(H, W)
        img_decompressed = np.clip(img_decompressed, 0.0, 1.0)
        
        print(f"Итерация декодирования {iteration + 1}/{num_iterations} завершена...")
        
    return img_decompressed
'''

def compress_color(img_rgb, thresh=0.0001, B_max=16, B_min=4, thresh_c=None):
    # Порог для цветоразностных каналов можно задать отдельно;
    # если не задан - используем тот же, что и для яркости (старое поведение).
    if thresh_c is None:
        thresh_c = thresh

    # RGB [0..255] -> YCbCr [0..255]
    ycbcr = rgb_to_ycbcr(img_rgb)

    # 2. Выделение каналов и нормировка в [0.0, 1.0]
    Y = ycbcr[:, :, 0] / 255.0
    Cb = subsample_2x(ycbcr[:, :, 1]) / 255.0  # Размер (128, 128)
    Cr = subsample_2x(ycbcr[:, :, 2]) / 255.0  # Размер (128, 128)

    t_Y = compress_quadtree(Y, thresh=thresh, B_max=B_max, B_min=B_min)
    t_Cb = compress_quadtree(Cb, thresh=thresh_c, B_max=B_max, B_min=B_min)
    t_Cr = compress_quadtree(Cr, thresh=thresh_c, B_max=B_max, B_min=B_min)

    return {
        't_Y': t_Y, 
        't_Cb': t_Cb, 
        't_Cr': t_Cr,
        'shape_Y': Y.shape,
        'shape_chroma': Cb.shape
    }

def decompress_color(color_data, target_shape=None, scale=1.0, num_iterations=20):
    shape_Y_orig = color_data.get('shape_Y', (512, 512))
    
    # Если передан target_shape (например, (2048, 2048)), считаем scale
    if target_shape is not None:
        scale = target_shape[0] / shape_Y_orig[0]
    else:
        target_shape = (
            int(round(shape_Y_orig[0] * scale)),
            int(round(shape_Y_orig[1] * scale))
        )

    shape_Y = target_shape
    shape_chroma = (target_shape[0] // 2, target_shape[1] // 2)

    # Масштабируем трансформации под новый размер
    t_Y = scale_transforms(color_data['t_Y'], scale)
    t_Cb = scale_transforms(color_data['t_Cb'], scale)
    t_Cr = scale_transforms(color_data['t_Cr'], scale)

    print(f"Восстановление Y ({shape_Y[1]}x{shape_Y[0]})...")
    Y = decompress_quadtree(t_Y, img_shape=shape_Y, num_iterations=num_iterations, default_val=1.0)

    print(f"Восстановление Cb ({shape_chroma[1]}x{shape_chroma[0]})...")
    Cb_sub = decompress_quadtree(t_Cb, img_shape=shape_chroma, num_iterations=num_iterations, default_val=0.5)
    
    print(f"Восстановление Cr ({shape_chroma[1]}x{shape_chroma[0]})...")
    Cr_sub = decompress_quadtree(t_Cr, img_shape=shape_chroma, num_iterations=num_iterations, default_val=0.5)

    H, W = shape_Y
    Cb = upsample_2x(Cb_sub, (H, W)) * 255.0
    Cr = upsample_2x(Cr_sub, (H, W)) * 255.0
    Y = Y * 255.0

    ycbcr = np.stack([Y, Cb, Cr], axis=2)
    return ycbcr_to_rgb(ycbcr)

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
