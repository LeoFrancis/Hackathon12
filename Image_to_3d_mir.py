import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from PIL import Image


MINIMUM_CLEARENCE = 20
GROUND_CLEARENCE = 40
buffer_size = 10
buffer_x = []
buffer_y = []

def lanGen(vector_data, min_distance):
    # Convert your vector data to a structured text description
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
    Resize((256, 256)),  # Resize to a fixed size expected by the model
    ToTensor(),  # Convert to a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
])

# Define the folder containing images
image_folder = 'Extreme'  # Change this to your folder path

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image in the folder
for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Unable to load image from path {img_path}. Skipping this file.")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)  # Convert the NumPy array to a PIL Image

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

    # Adjust y values
    #y_adjusted = np.array([v - 50 if (50 < v < 250) else v for v in y])

    # Filter points where y > 10 and z < 10
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
    print(f"Closest high point is detected as per image {image_file}: {closest_y}")

    x_low = x_filtered[~high_points_indices]
    y_low = y_filtered[~high_points_indices]
    z_low = z_filtered[~high_points_indices]

    z_final = z_high[z_high != 0]

    x_combined = np.append(x, x_high)
    y_combined = np.append(y, y_high)
    z_combined = np.append(z, z_high)

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

# Close all OpenCV windows
cv2.destroyAllWindows()
