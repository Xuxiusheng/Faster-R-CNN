from PIL import Image


def get_lines(train_lines, valid_lines):
    with open(train_lines, 'r') as f:
        train_lines = list(map(lambda x: x.strip(), f.readlines()))

    with open(valid_lines, 'r') as f:
        valid_lines = list(map(lambda x: x.strip(), f.readlines()))

    return train_lines, valid_lines

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width

def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image