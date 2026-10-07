import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from Colours import rgb_to_ycbcr, ycbcr_to_rgb, subsample_2x, upsample_2x
from Codec import compress, decompress, compress_color, decompress_color
from SaveAndLoad import save_pifs, load_pifs

if __name__ == "__main__":
    # 1. Загружаем RGB изображение
    img_pil = Image.open('astronaut_512.png').convert('RGB').resize((512, 512))
    img_rgb = np.array(img_pil)

    # 2. Сжатие
    color_transforms = compress_color(img_rgb)
    
    # 3. Сохранение в бинарный .pifs
    save_pifs("fern_color.pifs", color_transforms, img_shape=(512, 512))
    
    # 4. Загрузка из файла
    loaded_transforms, (w, h) = load_pifs("fern_color.pifs")
    
    # 5. Восстановление
    decompressed_rgb = decompress_color(loaded_transforms, img_shape=(w, h), num_iterations=20)

    # Отображение результатов
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Оригинал RGB")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Из файла .pifs")
    plt.imshow(decompressed_rgb)
    plt.axis("off")

    plt.show()