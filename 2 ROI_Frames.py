import cv2
import os

# Define the input and output folder paths
input_folder = 'Frames'            # Folder containing the captured frames
output_folder = 'Edited_Frames'    # Folder to save the cropped frames

# Make the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ROI (x, y, width, height) - adjust these values 
roi_x, roi_y, roi_w, roi_h = 110, 450, 500, 600  
# Process each image in the 'Frames' folder
def process_images(input_folder):
    # Loop through each file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            # Check if the image was successfully loaded
            if image is None:
                print(f"Error loading image: {filename}")
                continue

            # Crop the image based on the defined ROI
            cropped_image = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # Save the cropped image in the output folder
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, cropped_image)

    print("Processed all images.")

# Run the function to process the images
process_images(input_folder)
