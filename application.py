# import the necessary packages
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
from flask_cors import CORS
import io

# initialize our Flask application and the Keras model
application = flask.Flask(__name__)
CORS(application)

MODEL_PATH = 'models/kashir_cheez_transfer_model.h5'

model = tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()
graph  = tf.get_default_graph()
print('Model Loaded. Start Serving...')

#@application.before_first_request
#def load_img_model():
    #global model
    #model = tf.keras.models.load_model(MODEL_PATH)
    #model._make_predict_function()
    #global graph
    #graph = tf.get_default_graph()


def prepare_image(image, target):
    # resize the input image and preprocess it
    image = image.resize(target)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255

    # return the processed image
    return image


@application.route('/')
def index():
    return 'Kashir-Cheez Web Api'


@application.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(150, 150))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            index_to_class = ['Kangri', 'Namda', 'Pheran', 'Samavar']
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for i, pred in enumerate(preds[0]):
                r = {"label": index_to_class[i], "probability": float(pred)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    application.run(debug=True)
