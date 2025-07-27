import os
import shutil
import random
import pathlib

# --- Configuration ---
# IMPORTANT: Update this path to where you extracted your dataset.
# The error you see is because Python on Windows reads backslashes (\) as escape characters.
# To fix this, use a "raw string" by adding an 'r' before the quotes, like below.
SOURCE_DATA_DIR = r"C:\Users\Rakesh Roshan Allam\Downloads\Dataset (diseases)\3disease" 

# This is where the script will create the final structured data folder.
# A relative path like "data" is recommended and should work correctly from your project folder.
DEST_DATA_DIR = "data" 

# Define the ratio for splitting the data
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO is implicitly 0.1 (1.0 - 0.8 - 0.1)

# --- End of Configuration ---


def split_data():
    """
    Splits the data from the source directory into train, validation,
    and test sets in the destination directory.
    """
    print("Starting dataset preparation...")
    
    # Ensure the source directory exists
    source_path = pathlib.Path(SOURCE_DATA_DIR)
    if not source_path.exists():
        print(f"Error: Source directory not found at '{SOURCE_DATA_DIR}'")
        print("Please update the SOURCE_DATA_DIR variable in the script.")
        return

    # Get the class names from the folder names in the source directory
    class_names = [d.name for d in source_path.iterdir() if d.is_dir()]
    if not class_names:
        print(f"No disease folders found in '{SOURCE_DATA_DIR}'.")
        return
        
    print(f"Found classes: {class_names}")

    # Create destination directories
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            path = pathlib.Path(DEST_DATA_DIR) / split / class_name
            path.mkdir(parents=True, exist_ok=True)
            
    # Process each class folder
    for class_name in class_names:
        print(f"\nProcessing class: {class_name}")
        class_dir = source_path / class_name
        
        # Get all image files (supports common formats)
        image_files = list(class_dir.glob('*.jpg')) + \
                      list(class_dir.glob('*.jpeg')) + \
                      list(class_dir.glob('*.png'))
                      
        random.shuffle(image_files) # Shuffle for randomness
        
        total_images = len(image_files)
        train_split_index = int(total_images * TRAIN_RATIO)
        val_split_index = int(total_images * (TRAIN_RATIO + VAL_RATIO))
        
        # Assign images to splits
        train_images = image_files[:train_split_index]
        val_images = image_files[train_split_index:val_split_index]
        test_images = image_files[val_split_index:]
        
        print(f"  Total images: {total_images}")
        print(f"  Training: {len(train_images)}, Validation: {len(val_images)}, Testing: {len(test_images)}")
        
        # Function to copy files
        def copy_files(files, split_name):
            for f in files:
                dest_path = pathlib.Path(DEST_DATA_DIR) / split_name / class_name / f.name
                shutil.copy(str(f), str(dest_path))

        # Copy files to their new homes
        copy_files(train_images, 'train')
        copy_files(val_images, 'val')
        copy_files(test_images, 'test')

    print("\nDataset preparation complete!")
    print(f"Your structured dataset is ready in the '{DEST_DATA_DIR}' folder.")


if __name__ == "__main__":
    split_data()
