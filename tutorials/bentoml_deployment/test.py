import requests
from fastface.utils.vis import render_predictions
import imageio

url = "http://localhost:5000/detect"

payload={}
files=[
  ('image',('friends2.jpg',open('../../resources/friends2.jpg','rb'),'image/jpeg'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.json())

pretty_img = render_predictions(imageio.imread('../../resources/friends2.jpg'), response.json())

# show image
pretty_img.show()