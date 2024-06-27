import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from PIL import Image


GROUND_CLEARENCE = 20
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
image_folder = 'Backup_1'  # Change this to your folder path

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

    # Print all x_filtered and y_filtered values
    '''print('***********************************')
    print(f"All x values for {image_file}: {x}")
    print(f"All y values for {image_file}: {y}")
    print(f"All z values for {image_file}: {z}")'''

    # Adjust y values
    y_adjusted = np.array([v - 50 if (50 < v < 250) else v for v in y])

    # Filter points where y > 10 and z < 10
    filtered_indices = (y_adjusted > 10) & (z < 10)
    x_filtered = x[filtered_indices]
    y_filtered = y_adjusted[filtered_indices]
    z_filtered = z[filtered_indices]

    # Separate the points with height greater than GROUND_CLEARENCE
    high_points_indices = y_filtered > GROUND_CLEARENCE
    x_high = x_filtered[high_points_indices]
    y_high = y_filtered[high_points_indices]
    z_high = z_filtered[high_points_indices]

    x_low = x_filtered[~high_points_indices]
    y_low = y_filtered[~high_points_indices]
    z_low = z_filtered[~high_points_indices]

    z_final = z_high[z_high != 0]

    '''z_final = np.array
    # Get the top 5 y values
    for v in z_high:
        if ( z_high[v] > 0.0):
            z_final[v] = z_high[v]'''

    closest_distance = np.sort(z_final)[:]
    print(closest_distance)
    #final_array = z_final[:1]
    print(f"Closest high point is detected as per image {image_file}: {closest_distance[0:1]}")

    # Plot the filtered points in 2D and the original image
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Show the original image
    ax[0].imshow(img)
    ax[0].set_title(f"Original Image - {image_file}")
    ax[0].axis('off')

    # Plot x vs z
    scatter_low = ax[1].scatter(-x_low, z_low, c='blue', s=1)  # Invert x axis by multiplying by -1
    scatter_high = ax[1].scatter(-x_high, z_high, c='red', s=20)  # Larger sized points for y > 30
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Depth (Z)')
    ax[1].set_title(f'Filtered X vs Z Plot for {image_file}')
    ax[1].grid(True)

    # Set the axis limits
    ax[1].set_xlim(left=0, right=-w)  # Ensure the x-axis starts from 0 to the right
    ax[1].set_ylim(bottom=np.max(z_low), top=0)  # Ensure the y-axis (depth) starts from 0 to the bottom
    
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

    # Wait for a keypress to move to the next image
    key = input("Press Enter to continue to the next image or type 'exit' to quit: ")
    if key.lower() == 'exit':
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
