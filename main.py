from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import numpy as np
import io

# Define Pydantic model
class ImageData(BaseModel):
    filename: str
    content_type: str
    prediction: str


app = FastAPI()

client = MongoClient('mongodb://localhost:27017/')
db = client['mnist_db']
collection = db['predictions']


model = load_model("handwritten_text.h5")
class_labels = "0123456789"

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')
    img = img.resize((28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    image_data = ImageData(
        filename=file.filename,
        content_type=file.content_type,
        prediction=predicted_class
    )

    collection.insert_one(image_data.dict())

    return JSONResponse(content=image_data.dict())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


