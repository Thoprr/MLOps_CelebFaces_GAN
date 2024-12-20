from flask import Flask, send_file
from .generate_image import load_generator_model, generate_image

app = Flask(__name__)

@app.route('/image', methods=['GET'])
def get_image():

    """
    Generates an image using the generator model and returns it as a PNG file.

    The image is saved locally as 'image.png' and then sent as a response to the client.
    If an error occurs, it returns a JSON response with the error message and a 500 status code.

    Returns:
        Response: The generated PNG image file or a JSON error message with a 500 status code.
    """

    generate_image(generator, './application/image.png')

    try:
        return send_file('./image.png', mimetype='image/png')
    except Exception as e:
        return {"error": f"Une erreur est survenue : {str(e)}"}, 500

def run_app(model_api=""):

    """
    Loads a generator model and starts the Flask application.

    The generator model is loaded from the specified path or URL.
    The Flask app runs on host '0.0.0.0' and port 8000 with debug mode disabled.

    Args:
        model_api (str): Path or URL to the generator model to be loaded. Default is an empty string.

    Returns:
        None
    """

    global generator
    generator = load_generator_model(model_api)
    app.run(host='0.0.0.0', port=8000, debug=False)