import os
from PIL import Image
from torchvision import transforms

def scale_down_images_in_folders(main_folder, target_size=(70, 75)):
    """
    Scales down images in subfolders of the main folder and saves them to a new folder.

    Parameters:
    - main_folder (str): Path to the main folder containing subfolders.
    - target_size (tuple): Size to which the images will be resized (default is (100, 100)).
    """
    for data_type in ['train', 'test']:
        src_folder = os.path.join(main_folder, data_type)
        dest_folder = os.path.join(main_folder, f"{data_type}_scaled(70, 75)")

        # Create the destination folder if it doesn't exist
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Define the transformation to scale down images
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

        # Iterate through each subfolder in the source folder
        for subfolder in os.listdir(src_folder):
            subfolder_path = os.path.join(src_folder, subfolder)
            if os.path.isdir(subfolder_path):
                dest_subfolder = os.path.join(dest_folder, subfolder)
                os.makedirs(dest_subfolder, exist_ok=True)

                # Iterate through each file in the subfolder
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add more extensions if needed
                        # Load the image
                        img_path = os.path.join(subfolder_path, filename)
                        img = Image.open(img_path)

                        # Apply the transformation
                        scaled_img = transform(img)

                        # Save the scaled image to the destination folder
                        dest_path = os.path.join(dest_subfolder, filename)
                        scaled_img_pil = transforms.ToPILImage()(scaled_img)  # Convert back to PIL Image for saving
                        scaled_img_pil.save(dest_path)

                        print(f"Image {filename} scaled and saved to {dest_path}")

# Example usage:
main_folder = 'GG_4A'
scale_down_images_in_folders(main_folder, target_size=(70, 75))