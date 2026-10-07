import numpy as np
import struct

def save_pifs(filename, color_transforms, img_shape=(256, 256)):
    width, height = img_shape
    t_Y = color_transforms['t_Y']
    t_Cb = color_transforms['t_Cb']
    t_Cr = color_transforms['t_Cr']

    with open(filename, "wb") as f:
        header = struct.pack(">4sHH", b"PIFS", width, height)
        f.write(header)

        def write_channel(channel_blocks):
            f.write(struct.pack(">I", len(channel_blocks)))
            
            for t in channel_blocks:
                y, x, size = int(t['y']), int(t['x']), int(t['size'])
                j, k, s, o = int(t['j']), int(t['k']), float(t['s']), float(t['o'])

                s_norm = (s + 0.75) / 1.5
                s_q = int(np.clip(round(s_norm * 30.0), 0, 30))
                o_q = int(np.clip(round(o * 255.0), 0, 255))

                packed_val = ((j & 0x3FF) << 16) | ((k & 0x7) << 13) | ((s_q & 0x1F) << 8) | (o_q & 0xFF)

                f.write(struct.pack(">HHB", y, x, size))
                f.write(packed_val.to_bytes(4, byteorder="big"))

        write_channel(t_Y)
        write_channel(t_Cb)
        write_channel(t_Cr)

        actual_size = f.tell()
    print(
        f"Файл успешно сохранен: {filename} (Размер: {actual_size} байт)"
    )

def load_pifs(filename):
    with open(filename, "rb") as f:
        header = f.read(8)
        magic, width, height = struct.unpack(">4sHH", header)
        if magic != b"PIFS":
            raise ValueError("Файл не является цветным форматом PIFS!")

        def read_channel():
            count_bytes = f.read(4)
            count = struct.unpack(">I", count_bytes)[0]
            transforms = []

            for _ in range(count):
                y, x, size = struct.unpack(">HHB", f.read(5))
                chunk = f.read(4)
                packed_val = int.from_bytes(chunk, byteorder="big")

                j = (packed_val >> 16) & 0x3FF
                k = (packed_val >> 13) & 0x7
                s_q = (packed_val >> 8) & 0x1F
                o_q = packed_val & 0xFF

                s = (s_q / 30.0) * 1.5 - 0.75
                o = o_q / 255.0

                transforms.append({'y': y, 'x': x, 'size': size, 'j': j, 'k': k, 's': s, 'o': o})

            return transforms

        t_Y = read_channel()
        t_Cb = read_channel()
        t_Cr = read_channel()

    return {'t_Y': t_Y, 't_Cb': t_Cb, 't_Cr': t_Cr}, (width, height)