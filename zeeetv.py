import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Load the pre-trained SSD model (SSD300 with VGG16 backbone)
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the transformation for input image (to match the SSD model input size)
transform = T.Compose([
    T.ToTensor(),  # Convert the image to a Tensor
])

# COCO Classes (80 object categories)
coco_classes = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
    11: 'fire hydrant', 13: 'dog', 14: 'horse', 15: 'sheep', 16: 'cow',
    17: 'Cat', 18: 'bear', 19: 'Horse', 20: 'giraffe', 
    21: 'backpack', 22: 'umbrella', 23: 'handbag', 24: 'Zebra', 
    25: 'suitcase',26:'snowboard',27: 'frisbee', 28: 'skis', 29: 'baseball bat', 
    30: 'sports ball', 31: 'kite', 32: 'Tie', 33: 'baseball glove', 
    34: 'skateboard', 35: 'surfboard', 36: 'tennis racket', 37: 'dining table', 
    38: 'wine glass', 39: 'cup', 40: 'fork', 41: 'knife', 42: 'spoon', 
    43: 'bowl', 44: 'Bottle', 45: 'apple', 46: 'sandwich', 47: 'orange', 
    48: 'broccoli', 49: 'carrot', 50: 'hot dog', 51: 'pizza', 52: 'banana', 
    53: 'apple', 54: 'chair', 55: 'couch', 56: 'potted plant', 
    57: 'bed', 58: 'tv', 59: 'toilet', 60: 'donut', 
    61: 'laptop', 62: 'mouse', 63: 'remote', 64: 'keyboard', 
    65: 'cake', 67: 'microwave', 68: 'oven', 69: 'toaster', 
    70: 'sink', 71: 'refrigerator', 72: 'book', 73: 'clock', 
    74: 'vase', 75: 'scissors', 76: 'teddy bear', 77: 'mobile phone', 
    78: 'toothbrush'
}

# Function to classify an image from file (for object detection)
def detect_objects(image):
    # Apply the transformation
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Run the model (no gradients needed during inference)
    with torch.no_grad():
        prediction = model(img_tensor)
    
    return prediction

# Function to display the object name on the image (bounding boxes + labels)
def display_labels_on_image(img, prediction):
    # Convert image to numpy array
    img_np = np.array(img)
    
    # Get prediction details: boxes, labels, and scores
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Iterate through the detections and draw bounding boxes
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold for object detection
            box = boxes[i]
            label = coco_classes.get(labels[i], "Unknown")
            score = scores[i]
            
            # Draw rectangle around the detected object
            cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            
            # Put the label and score on top of the box
            cv2.putText(img_np, f'{label}: {score:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return img_np

# Main code block for object detection with webcam
def webcam_classification():
    cap = cv2.VideoCapture(0)  # Use the default webcam

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convert the frame to PIL Image for model input
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform object detection on the frame
        prediction = detect_objects(pil_img)

        # Display the frame with bounding boxes and labels
        img_with_labels = display_labels_on_image(frame, prediction)

        # Show the frame with labels using OpenCV
        cv2.imshow("Detected Objects", img_with_labels)

        # Stop the webcam when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# GUI for Image Classification or Webcam Classification
def start_gui():
    def on_image_classification():
        file_path = filedialog.askopenfilename(title="Select Image for Detection", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            try:
                img = Image.open(file_path)
                prediction = detect_objects(img)
                img_with_labels = display_labels_on_image(np.array(img), prediction)
                plt.imshow(img_with_labels)
                plt.axis('off')
                plt.show()
            except Exception as e:
                messagebox.showerror("Error", f"Error in processing image: {e}")

    def on_webcam_classification():
        webcam_classification()

    # Create the main window
    window = tk.Tk()
    window.title("Object Detection")

    # Set up window size and position
    window.geometry("400x200")

    # Add title label
    title_label = tk.Label(window, text="Image Classification", font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    # Add buttons
    btn_image = tk.Button(window, text="Select Image for Detection", command=on_image_classification, width=30, height=2)
    btn_image.pack(pady=10)

    btn_webcam = tk.Button(window, text="Webcam Classification", command=on_webcam_classification, width=30, height=2)
    btn_webcam.pack(pady=10)

    # Start the GUI event loop
    window.mainloop()

if __name__ == "__main__":
    start_gui()
