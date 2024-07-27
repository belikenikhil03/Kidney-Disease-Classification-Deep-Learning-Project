import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = os.path.join("artifacts", "training", "model.h5")
    
    def predict(self):
        # load model
        try:
            model = load_model(self.model_path)
            print("Model loaded successfully from:", self.model_path)
            print("Model output shape:", model.output_shape)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return [{"error": "Model could not be loaded"}]
        
        # load and preprocess the image
        try:
            test_image = image.load_img(self.filename, target_size=(224,224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0  # Normalize the image
            print("Image shape:", test_image.shape)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return [{"error": "Image could not be processed"}]
        
        # make prediction
        result = model.predict(test_image)
        print("Raw prediction results:", result)
        
        class_index = np.argmax(result, axis=1)[0]
        
        # map class index to label
        class_labels = ['Normal', 'Tumor', 'Cyst', 'Stone']
        prediction = class_labels[class_index]
        
        # get confidence score
        confidence = np.max(result) * 100
        
        print(f"Predicted class index: {class_index}")
        print(f"Predicted class: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        
        return [{"image": prediction, "confidence": f"{confidence:.2f}%", "raw_results": result.tolist()}]

# Test the pipeline
pipeline = PredictionPipeline("path_to_your_image.jpg")
result = pipeline.predict()
print(result)