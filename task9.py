import cv2
import torch
import time
from picamera2 import Picamera2
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Define the custom model class based on the training architecture.
class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet, self).__init__()
        # Use ResNet18 as base model, and replace the classifier layer (fc) with custom classifier
        self.resnet = models.resnet18(pretrained=False)  # Using ResNet-18 as the base
        # Modify the final fully connected layer for the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Initialize the custom model
model = CustomResNet(num_classes=2)  # Assuming you have 2 classes (Luke and Rory)

# Load the saved state dict into the model
try:
    model.load_state_dict(torch.load('lab8/best_model_ce.pth'))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize the camera
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
camera.start()
time.sleep(0.1)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Image transformation for input to the model (assuming model expects 224x224 images)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Labels mapping (assuming 0 is "Luke" and 1 is "Rory")
labels = {0: "Luke", 1: "Rory"}

img_count = 0
while True:
    # Capture frame from camera
    frame = camera.capture_array()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face_image = frame[y:y + h, x:x + w]

        # Convert the cropped face to a PIL image and apply transformations
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        input_image = transform(pil_image).unsqueeze(0)  # Add batch dimension

        # Use the model to classify the face
        with torch.no_grad():  # Don't need gradients for inference
            output = model(input_image)
            _, predicted = torch.max(output, 1)  # Get the class with the highest score
            label = labels[predicted.item()]  # Get the name based on the prediction

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the image with face bounding boxes and the classified label
    cv2.imshow("Face Detection and Classification", frame)

    # Save image when spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite(f"task9img{img_count}.jpg", frame)
        img_count += 1

# Clean up when exiting
camera.stop()
cv2.destroyAllWindows()
