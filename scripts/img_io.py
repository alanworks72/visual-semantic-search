from PIL import Image, ImageOps


def load_images(paths):
    images = []
    
    for path in paths:
        with Image.open(path) as img:
            images.append(ImageOps.exif_transpose(img).convert('RGB'))
    
    return images