import os

# Define the directory to search for .txt files
directory = "/Users/nguyenconghung/Downloads/yolov8/DataMatrix/valid/labels"  # Change this to the directory where your .txt files are located

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # Check if the file is a .txt file
        file_path = os.path.join(directory, filename)

        # Read and modify the file content
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Process each line
        modified_lines = []
        for line in lines:
            parts = line.split()
            if parts and parts[0] == "0":  # Change only the first element if it's "0"
                parts[0] = "2"
            modified_lines.append(" ".join(parts) + "\n")

        # Write back the modified content to the same file
        with open(file_path, "w") as file:
            file.writelines(modified_lines)

        print(f"Updated: {filename}")

print("Processing complete.")
