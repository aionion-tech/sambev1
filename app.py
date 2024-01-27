import os
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import requests
import torch
from fastsam import FastSAM, FastSAMPrompt
from flask_cors import CORS

model = FastSAM('./FastSAM-x.pt')


app = Flask(__name__)
CORS(app)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


@app.route('/process', methods=['POST'])
def process_image():
    image_url = request.json['imageUrl']
    bbox = request.json['bbox']
    # centerX = int(request.json['centerPoint'][0])
    # centerY = int(request.json['centerPoint'][1])

    # print(centerX, centerY)
    image = Image.open(io.BytesIO(requests.get(image_url).content))
    # image = np.array(image)
    img_width = image.size[0]
    img_height = image.size[1]
    everything_results = model(
        image, device=DEVICE, retina_masks=True, imgsz=[img_width, img_height], conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(
        image, everything_results, device=DEVICE)

    ann = prompt_process.everything_prompt()

    print(bbox, img_width,  img_height)
    # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    ann = prompt_process.box_prompt(bboxes=[bbox])
    # ann = prompt_process.box_prompt(
    #     bboxes=[[187.95735994556588, 369.905636317266, 359.9183488319346, 529.8648304004081]])

    print('SHAPEEEE', ann.shape)
    # ann = prompt_process.text_prompt(text='a photo of a cat')

    # point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    # ann = prompt_process.point_prompt(
    #     points=[[centerX, centerY]], pointlabel=[1])

    return jsonify({
        "result": ann.tolist()
    })


if __name__ == '__main__':
    app.run(debug=True)
