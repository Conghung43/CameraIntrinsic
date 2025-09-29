import os
import json

def convert_json_to_yolov8(json_path, output_dir):
    """
    Convert a JSON file to YOLOv8 TXT format and save the output.
    
    Args:
        json_path (str): Path to the JSON file.
        output_dir (str): Directory to save the YOLOv8 TXT files.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Image dimensions
    image_width = data["Background"]["Width"]
    image_height = data["Background"]["Height"]

    # Prepare YOLOv8 format output
    yolo_data = []

    # Iterate over objects
    for obj in data["Objects"]:
        if obj["Layers"]:
            # Extract bounding box points
            if 'Points' in obj["Layers"][0]["Shape"]:
                points = obj["Layers"][0]["Shape"]["Points"]
                try:
                    x_coords = [int(float(point.split(',')[0])) for point in points]
                    y_coords = [int(float(point.split(',')[1])) for point in points]

                    # Calculate bounding box properties
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                except:
                    print()

            else:
                PS0 = obj["Layers"][0]["Shape"]["P0"]
                PS1 = obj["Layers"][0]["Shape"]["P1"]
                P0 = PS0.split(',') 
                P1 = PS1.split(',')

                # Calculate bounding box properties
                x_min = int(float(P0[0]))
                x_max = int(float(P1[0]))
                y_min = int(float(P0[1]))
                y_max = int(float(P1[1]))

            # Calculate YOLOv8 format values
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            box_width = (x_max - x_min) / image_width
            box_height = (y_max - y_min) / image_height

            # Use class ID 0 for this dataset (assuming a single class "DataMatrixCode")
            class_id = 0
            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Save the YOLOv8 format data to a TXT file
    output_filename = os.path.splitext(os.path.basename(json_path))[0].split('.')[0] + '.txt'
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as output_file:
        output_file.write("\n".join(yolo_data))

    print(f"Converted: {json_path} -> {output_path}")

# Main function
def process_directory(json_dir, output_dir):
    """
    Process all JSON files in a directory and convert them to YOLOv8 format.

    Args:
        json_dir (str): Directory containing JSON files.
        output_dir (str): Directory to save the YOLOv8 TXT files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            convert_json_to_yolov8(json_path, output_dir)

# Example usage
# Replace 'path_to_json_directory' and 'path_to_output_directory' with your paths
# process_directory('path_to_json_directory', 'path_to_output_directory')

# Paths for input directory and output directory
# input_json_dir = '/Users/nguyenconghung/Downloads/Images-1'
# output_json_dir = '/Users/nguyenconghung/Downloads/yolov8'

# # Perform the conversion for all JSON files
# process_directory(input_json_dir, output_json_dir)

# print(f"All JSON files in {input_json_dir} have been converted and saved in {output_json_dir}.")

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')  # Replace with your model's path if it's not in the current directory

# Export the model to ONNX
model.export(format="onnx")