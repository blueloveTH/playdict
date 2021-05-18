import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw

def random_color(start=0, end=255):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    return (red, green, blue)

def create_bg(width, height):
    img = Image.new('RGBA', (width, height), random_color())
    if random.random() < 0.3:
        create_noise_dots(img, random_color(), number=random.randint(height//4, height))
    if random.random() < 0.3:
        for _ in range(random.randint(0, height//16)):
            create_noise_line(img, random_color())
    return img

def create_noise_dots(image, color, number):
    draw = Draw(image)
    w, h = image.size
    while number:
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=random.randint(2,4))
        number -= 1
    return image


def create_noise_curve(image, color):
    w, h = image.size
    x1 = random.randint(0, int(w / 5))                      # [0, 12]
    x2 = random.randint(w - int(w / 5), w)                  # [42, 64]
    y1 = random.randint(int(h / 5), h - int(h / 5))         # [12, 42]
    y2 = random.randint(y1, h - int(h / 5))                 # [[12, 42], 42]
    points = [x1, y1, x2, y2]
    end = random.randint(160, 200)
    start = random.randint(0, 20)
    Draw(image).arc(points, start, end, fill=color)
    return image


def create_noise_line(image, color):
    w, h = image.size
    x1 = random.randint(0, int(w / 5))                      # [0, 12]
    x2 = random.randint(w - int(w / 5), w)                  # [42, 64]
    y1 = random.randint(int(h / 5), h - int(h / 5))         # [12, 42]
    y2 = random.randint(int(h / 5), h - int(h / 5))         # [[12, 42], 42]
    points = [x1, y1, x2, y2]
    Draw(image).line(points, fill=color)
    return image