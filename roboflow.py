from roboflow import Roboflow
rf = Roboflow(api_key="VKi8pA7O3w2d15dgVoDs")
project = rf.workspace().project("yolo-waste-detection")
model = project.version(1).model

# infer on a local image
# print(model.predict("image.png", confidence=40, overlap=30).json())

# visualize your prediction
print(model.predict("image.jpg", confidence=40, overlap=30).save("pred.jpg"))

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())