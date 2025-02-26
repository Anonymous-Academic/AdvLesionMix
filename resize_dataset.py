import os
from PIL import Image


source_dir = ''
target_dir = ''

def resize_image(source_path, target_path, size=(256, 256)):
    """
    Resize the image to the specified size and save it to the target path.
    """
    with Image.open(source_path) as img:
        # Resize image with highest quality
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        img_resized.save(target_path, quality=100)

def process_directory(source_dir, target_dir):
    """
    Process all images in source_dir and save resized versions to target_dir.
    """
    for root, _, files in os.walk(source_dir):
        # Map source directory to target directory
        relative_path = os.path.relpath(root, source_dir)
        target_root = os.path.join(target_dir, relative_path)
        
        # Ensure the target directory exists
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_root, file)
                # Resize and save the image
                resize_image(source_path, target_path)

# Run the resizing process
process_directory(source_dir, target_dir)
print("All images have been resized and saved.")
