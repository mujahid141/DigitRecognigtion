# your_app/views.py
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render  # For rendering templates
from PIL import Image
import os

def predict_digit(request):
    """
    Django view function to handle digit prediction requests.

    - Renders the HTML template for the user interface (GET request).
    - Processes image uploads and predicts the digit (POST request).

    Returns an appropriate HTTP response depending on the request type.
    """

    if request.method == 'GET':
        # Render the HTML template for user input
        return render(request, 'app/predict_digit.html')  # Replace with your template name

    elif request.method == 'POST':
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'Missing image file in request.'}, status=400)

        try:
            # Define the paths to the model architecture and weights
            model_architecture_path = os.path.join('app', 'model', 'config.json')
            model_weights_path = os.path.join('app', 'model', 'model.weights.h5')

            # Load the model architecture from the JSON file
            with open(model_architecture_path, 'r') as json_file:
                model_json = json_file.read()
            new_model = model_from_json(model_json, custom_objects={'softmax_v2': tf.nn.softmax})
            
            # Load the weights into the model
            new_model.load_weights(model_weights_path)

            # Get the image data from the uploaded file
            image_file = request.FILES['image']
            image = Image.open(image_file)
            image = image.convert('L')  # Convert to grayscale
            image = image.resize((28, 28))  # Resize to the model's expected input size

            # Convert the image to a numpy array
            image_data = np.array(image, dtype=np.float32)

            # Normalize the image data
            image_data = image_data / 255.0

            # Reshape the image to match the input shape of the model
            image_data = image_data.reshape(1, 28, 28, 1)

            # Make the prediction
            predictions = new_model.predict(image_data)
            predicted_digit = np.argmax(predictions[0])

            return JsonResponse({'predicted_digit': int(predicted_digit), 'message': 'Prediction successful!'})

        except (ValueError, Exception) as e:
            print(f"Error during prediction: {e}")
            return JsonResponse({'error': str(e)}, status=400)  # Bad request

    else:
        return JsonResponse({'error': 'Only GET or POST requests are allowed.'}, status=405)
