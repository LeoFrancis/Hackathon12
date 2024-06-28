import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from PIL import Image
import pyttsx3
import base64
import requests
from openai import OpenAI

api_key = ""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

MINIMUM_CLEARENCE = 20
GROUND_CLEARENCE = 40
buffer_size = 10
buffer_x = []
buffer_y = []

engine = pyttsx3.init()

voices = engine.getProperty('voices')
for voice in voices:
    print(f"Voice: {voice.name}, ID: {voice.id}")

engine.setProperty('voice', voices[1].id) 
engine.setProperty('rate', 160)    
engine.setProperty('volume', 0.9) 

def lanGen(vector_data, min_distance):
    return f"Observing obstruction. Shortest distance to high object: {min_distance:.2f} meters."

# Check the current working directory
current_working_directory = os.getcwd()
print(f"Current working directory: {current_working_directory}")

# Load the MiDaS model and move it to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
midas.eval()

# Define the necessary transformations
midas_transforms = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the folder containing images
image_folder = 'POC'
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Function to plot a single large arrow
def plot_large_arrow(ax, start, direction, length, color='red', arrow_size=100, arrow_width=100):
    ax.quiver(
        start[0], start[1], start[2],
        direction[0], direction[1], direction[2],
        length=length,
        color=color,
        arrow_length_ratio=0.3,  # Control arrow head size
        lw=arrow_width
    )

# Process each image in the folder
for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    base64_image = encode_image(img_path)
    client = OpenAI(api_key=api_key)
    img = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Unable to load image from path {img_path}. Skipping this file.")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    # Apply MiDaS transforms
    input_batch = midas_transforms(img_pil).unsqueeze(0).to(device)

    # Perform depth estimation
    with torch.no_grad():
        depth_map = midas(input_batch)

    # Convert depth map to numpy array
    depth_map = depth_map.squeeze().cpu().numpy()

    # Create a mesh grid of coordinates
    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = depth_map.flatten()

    filtered_indices = (z > 10) & (y < 20)
    x_filtered = x[filtered_indices]
    y_filtered = y[filtered_indices]
    z_filtered = z[filtered_indices]

    # Separate the points with height greater than GROUND_CLEARENCE
    high_points_indices = z_filtered > MINIMUM_CLEARENCE
    x_high = x_filtered[high_points_indices]
    y_high = y_filtered[high_points_indices]
    z_high = z_filtered[high_points_indices]

    # Identifying the relevant indices
    Relevant_indices = (70 < x_high) & (x_high < 130) & (y_high < 20) & (z_high > GROUND_CLEARENCE) 
    x_relevant = x_high[Relevant_indices]
    y_relevant = y_high[Relevant_indices]
    z_relevant = z_high[Relevant_indices]
    sorted_indices = np.argsort(z_relevant)
    closest_distance_index = sorted_indices[0:1]
    closest_y = y_relevant[closest_distance_index]
    prompt = f"Closest high point detected {closest_y} m from vehicle"
    print(prompt)

    # Chatgpt prompting
    response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.2,
    messages=[
        {
            "role": "system",
            "content": "You are a driving assistant for off road scenarios. Your job is to prevent collisions and scratches on the vehicle",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt + ". The ground clearance of our vehicle is not sufficient. Print the terrain type and if this path drivable?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": "{\"terrain_type\": 'dirt track/unpaved road', \"advice\": 'Obstacle is higher than ground clearance. Risk of collision and damage to the car'\}",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt + ". The ground clearance of our vehicle is not sufficient. Print the terrain type and if this path drivable?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ],
    max_tokens=300,
    )
    print(response.choices[0].message.content)

    # For text to audio
    if closest_y:
        engine.say("Careful!, Possible body collision ahead due to terrain not enough ground clearence")
        engine.runAndWait()

    x_low = x_filtered[~high_points_indices]
    y_low = y_filtered[~high_points_indices]
    z_low = z_filtered[~high_points_indices]

    z_final = z_high[z_high != 0]

    x_combined = np.append(x, x_high)
    y_combined = np.append(y, y_high)
    z_combined = np.append(z, z_high)

    filtered_indices_plot = (x > 100) & (x < 150) & (y > -3) & (y < 15) & (z > 0) & (z < 100) 
    x_plot = x[filtered_indices_plot]
    y_plot = y[filtered_indices_plot]
    z_plot = z[filtered_indices_plot]

    print(f"Original Image - {image_file}")

    fig = plt.figure(figsize=(15, 7))

    # Plot the original image
    ax1 = fig.add_subplot(121)
    ax1.imshow(img)
    ax1.set_title(f"Original Image - {image_file}")
    ax1.axis('off')

    # Plot the 3D points
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x, y, z, c=z, cmap='Greens', label='Existing')
    ax2.scatter(x_high, y_high, z_high, c=z_high, cmap='Reds', label='High')
    ax2.scatter(x_plot, y_plot, z_plot)
    
    # Add one large arrow to represent direction
    arrow_start = (120, -20, 0)
    arrow_direction = (0, 1, 0)  # Arrow pointing in the z direction
    arrow_length = 10
    plot_large_arrow(ax2, arrow_start, arrow_direction, arrow_length, color='blue', arrow_size=20, arrow_width=5)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.title('3D Depth Map')
    plt.legend()
    plt.show()

    # Wait for a keypress to move to the next image
    key = input("Press Enter to continue to the next image or type 'exit' to quit: ")
    if key.lower() == 'exit':
        break

cv2.destroyAllWindows()
