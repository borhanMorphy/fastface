from PIL import Image, ImageDraw, ImageColor
import imageio
import random

color = random.choice(list(ImageColor.colormap.keys()))

img = imageio.imread("/home/morphy/Desktop/download.jpeg")

h,w = img.shape[:2]
x1,y1 = (0,0)
x2,y2 = (w//2,h//2)
shift = 5

img_pil = Image.fromarray(img)

for i in range(3):
    x1 += i*shift
    y1 += i*shift
    x2 += i*shift
    y2 += i*shift

    # create line image
    ImageDraw.Draw(img_pil).rectangle([(x1,y1),(x2,y2)], outline=color, width=3)

img_pil.show()