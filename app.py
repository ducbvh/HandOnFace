from flask import Flask, request, render_template
import numpy as np
import cv2, os
import tensorflow as tf
# import torch
from utils_model.my_model import Generator
import matplotlib.pyplot as plt

OUTPUT_FOLDER = os.path.join('static', 'output_image')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = OUTPUT_FOLDER

# load machine learning model

def load_img(input_image):
    input_image = tf.image.decode_png(input_image, channels= 3)
    #normalize
    input_image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1
    #resize
    input_image = tf.image.resize(input_image, (256, 256), method= 'bilinear')
    return input_image



def deocclude(input_image):
    #model
    gen_model = Generator(train_attention=False)
    gen_model.build((None, 256,256,3))
    status = gen_model.load_weights(r"./model_weight/generator2.h5", by_name=True)
    input_image = tf.expand_dims(input_image, axis=0)
    # output
    output_image = gen_model(input_image, training=True)
    output_image = output_image[0].numpy()#*0.5+0.5
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    return output_image


def occlude(image):
    pass
    # return output_image

model = {
    'deocclusion': deocclude,
    'occlusion': occlude
}

@app.route('/', methods=['GET','POST'])

def home():

    if request.method == 'POST':
        operation = request.form['operation']
        file = request.files['image']  # get uploaded image file

        img_data = file.read()  # read file contents

        img = load_img(img_data)

        new_file_name =  os.path.join(app.config['UPLOAD_FOLDER'], f'input.jpg')
        show_image = img.numpy()*127+127
        show_image =  cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(new_file_name,show_image)
        
        #Generate imge
        output_image = model[operation](img)

        file_name =  os.path.join(app.config['UPLOAD_FOLDER'], f'output.jpg')
        output_image = cv2.resize(output_image,(256,256),interpolation = cv2.INTER_LINEAR )


        cv2.imwrite(file_name,(output_image+1)*127)
        if file_name != None:
            return render_template("index.html", output  = file_name, input=new_file_name)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)