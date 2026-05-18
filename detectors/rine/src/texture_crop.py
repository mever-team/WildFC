import torch
import numpy as np
from torchvision.transforms import CenterCrop, TenCrop, Compose, ToTensor
from PIL import Image
from scipy.stats import entropy

def histogram_entropy_response(image):
    """
    Calculates the entropy of the image.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True) 
    prob_dist = histogram / histogram.sum()
    entr = entropy(prob_dist + 1e-7, base=2)   

    return entr


def texture_crop(image, stride = 224, window_size = [224, 224], drop = False):
    cropped_images = []
    images = []
    x, y =0, 0 

    for y in range(0, image.height - window_size[1] + 1, stride):
        for x in range(0, image.width - window_size[0] + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size[0], y + window_size[1])))
    
    if not drop:
        x = x + stride
        y = y + stride

        if x != image.width and x + window_size[0] > image.width and y + window_size[1] == image.height:
            for y in range(0, image.height - window_size[1] + 1, stride):
                cropped_images.append(image.crop((image.width - window_size[0], y, image.width, y + window_size[1])))
        elif y != image.height and x + window_size[0] == image.width and y + window_size[1] > image.height:
            for x in range(0, image.width - window_size[0] + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size[1], x + window_size[0], image.height)))
        elif x != image.width and y != image.height and x + window_size[0] > image.width and y + window_size[1] > image.height:
            for x in range(0, image.width - window_size[0] + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size[1], x + window_size[0], image.height)))
            for y in range(0, image.height - window_size[1] + 1, stride):
                cropped_images.append(image.crop((image.width - window_size[0], y, image.width, y + window_size[1])))
            cropped_images.append(image.crop((image.width - window_size[0], image.height - window_size[1], image.width, image.height)))

    for crop in cropped_images:
        metric = histogram_entropy_response(np.array(crop.convert('L')))
        images.append((crop, metric))

    images.sort(key=lambda x: x[1], reverse=True)
    texture_images = [img for img, _ in images[:10]]

    while len(texture_images) < 10:
        texture_images.append(texture_images[(len(texture_images) - 1) % len(images)]) 
    
    return texture_images


def tcrop(image, stride, window_size, drop):
    images = texturecrop(image, stride = 224, window_size = [224, 224], drop = False)
    img = random.choice(images)[0] 
    return img
