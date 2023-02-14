from flask import Flask, render_template, request, redirect, url_for
import base64
import os

from net import Net
import torch

app = Flask(__name__)

model = torch.load("net.pth")
model.eval()

# @app.route('/')
# def index():
#     return render_template('index.html')
@app.route('/')
def ind():
    return render_template('index.html')
@app.route('/guess/<guess>x')
def index(guess=''):
    return render_template('index.html',guess=guess)


from PIL import Image
import io
from torchvision import transforms
import matplotlib.pyplot as plt

@app.route('/submit', methods=['POST'])
def submit():
    img_data = request.form['img_data']
    img_data = img_data[22:]  # remove prefix data:image/png;base64,
    img_binary = base64.b64decode(img_data)

    # Open image using Pillow
    image = Image.open(io.BytesIO(img_binary))

    # Resize image to 28x28 using Pillow
    image = image.convert('RGB')
    resized_image = image.resize((28, 28))
    convert_tensor = transforms.ToTensor()
    tens= convert_tensor(resized_image)
    # plt.imshow(tens[0].view(28,28))
    # plt.show()

    guess = torch.argmax(model(tens[0].view(-1,28*28))[0]).item()
    print(guess)
    print(f'/guess/{guess}x')
    return redirect(f'/guess/{guess}x')

if __name__ == '__main__':
    app.run(debug=True)