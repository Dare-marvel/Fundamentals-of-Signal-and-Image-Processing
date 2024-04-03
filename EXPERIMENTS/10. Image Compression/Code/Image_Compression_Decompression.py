from PIL import Image
import os
import numpy as np

def compress_image(input_image_path, compressed_image_path, quality=80):
    original_image = Image.open(input_image_path)
    original_size = os.path.getsize(input_image_path)

    original_image.save(compressed_image_path, 'JPEG', quality=quality, optimize=True)
    compressed_size = os.path.getsize(compressed_image_path)

    return original_size, compressed_size

def decompress_image(compressed_image_path, decompressed_image_path):
    compressed_image = Image.open(compressed_image_path)
    compressed_image.save(decompressed_image_path, 'JPEG', quality=100)
    decompressed_size = os.path.getsize(decompressed_image_path)

    return decompressed_size

def calculate_metrics(original_image_path, compressed_image_path):
    original_image = Image.open(original_image_path).convert('RGB')
    compressed_image = Image.open(compressed_image_path).convert('RGB')

    original_image_array = np.array(original_image)
    compressed_image_array = np.array(compressed_image)

    mse = np.mean((original_image_array - compressed_image_array) ** 2)
    max_pixel_value = 255
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    original_image_size = os.path.getsize(original_image_path)
    compressed_image_size = os.path.getsize(compressed_image_path)
    bpp_original = (original_image_size * 8) / (original_image.size[0] * original_image.size[1])
    bpp_compressed = (compressed_image_size * 8) / (compressed_image.size[0] * compressed_image.size[1])

    return mse, psnr, bpp_original, bpp_compressed

# Example usage
input_image_path = 'B:\\Image_Comp_Decomp\\pulp_fiction.jpg'
compressed_image_path = 'compressed_image.jpg'
decompressed_image_path = 'decompressed_image.jpg'

original_size, compressed_size = compress_image(input_image_path, compressed_image_path)
decompressed_size = decompress_image(compressed_image_path, decompressed_image_path)

print(f'Original image size: {original_size} bytes')
print(f'Compressed image size: {compressed_size} bytes')
print(f'Decompressed image size: {decompressed_size} bytes')

mse, psnr, bpp_original, bpp_compressed = calculate_metrics(input_image_path, compressed_image_path)
print(f'MSE: {mse}')
print(f'PSNR: {psnr} dB')
print(f'BPP (Original): {bpp_original}')
print(f'BPP (Compressed): {bpp_compressed}')