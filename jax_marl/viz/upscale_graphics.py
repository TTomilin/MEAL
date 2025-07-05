#!/usr/bin/env python3
"""
Upscale the PNG files in the visualization/data/graphics directory to make them less pixelated.
This script will:
1. Load each PNG file
2. Upscale it to 2x or 4x the original resolution using high-quality interpolation
3. Save the upscaled version, replacing the original file
4. Load the corresponding JSON file (if it exists)
5. Update the frame positions and sizes to reflect the new resolution
6. Save the updated JSON file, replacing the original file
"""

import os
import json
from PIL import Image

# Directory containing the graphics files
GRAPHICS_DIR = os.path.join('jax_marl', 'viz', 'visualization', 'data', 'graphics')

# Scale factor for upscaling (2x, 3x, or 4x)
SCALE_FACTOR = 4

def upscale_image(image_path, scale_factor):
    """
    Upscale an image by the given scale factor using high-quality interpolation.
    
    Args:
        image_path (str): Path to the image file
        scale_factor (int): Factor by which to upscale the image (e.g., 2 for 2x)
        
    Returns:
        PIL.Image: The upscaled image
    """
    # Load the image
    image = Image.open(image_path)
    
    # Get the original size
    width, height = image.size
    
    # Calculate the new size
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    # Upscale the image using high-quality interpolation
    upscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return upscaled_image

def update_json(json_path, scale_factor):
    """
    Update a JSON file to reflect the new frame sizes and positions after upscaling.
    
    Args:
        json_path (str): Path to the JSON file
        scale_factor (int): Factor by which the images were upscaled
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's the format of soups.json
    if "textures" in data:
        # Update the scale
        data["textures"][0]["scale"] = str(float(data["textures"][0]["scale"]) * scale_factor)
        
        # Update the frame positions and sizes
        for frame in data["textures"][0]["frames"]:
            frame["frame"]["x"] *= scale_factor
            frame["frame"]["y"] *= scale_factor
            frame["frame"]["w"] *= scale_factor
            frame["frame"]["h"] *= scale_factor
            
            # Update the source size
            frame["sourceSize"]["w"] *= scale_factor
            frame["sourceSize"]["h"] *= scale_factor
    else:
        # Update the meta size
        if "meta" in data and "size" in data["meta"]:
            data["meta"]["size"]["w"] *= scale_factor
            data["meta"]["size"]["h"] *= scale_factor
        
        # Update the scale
        if "meta" in data and "scale" in data["meta"]:
            data["meta"]["scale"] = str(float(data["meta"]["scale"]) * scale_factor)
        
        # Update the frame positions and sizes
        for frame_name, frame_data in data["frames"].items():
            frame_data["frame"]["x"] *= scale_factor
            frame_data["frame"]["y"] *= scale_factor
            frame_data["frame"]["w"] *= scale_factor
            frame_data["frame"]["h"] *= scale_factor
            
            # Update the source size
            frame_data["sourceSize"]["w"] *= scale_factor
            frame_data["sourceSize"]["h"] *= scale_factor
            
            # Update the sprite source size
            if "spriteSourceSize" in frame_data:
                frame_data["spriteSourceSize"]["x"] *= scale_factor
                frame_data["spriteSourceSize"]["y"] *= scale_factor
                frame_data["spriteSourceSize"]["w"] *= scale_factor
                frame_data["spriteSourceSize"]["h"] *= scale_factor
    
    # Save the updated JSON file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Get all PNG files in the graphics directory
    png_files = [f for f in os.listdir(GRAPHICS_DIR) if f.endswith('.png')]
    
    # Process each PNG file
    for png_file in png_files:
        png_path = os.path.join(GRAPHICS_DIR, png_file)
        
        # Upscale the image
        print(f"Upscaling {png_file}...")
        upscaled_image = upscale_image(png_path, SCALE_FACTOR)
        
        # Save the upscaled image, replacing the original
        upscaled_image.save(png_path)
        
        # Check if there's a corresponding JSON file
        json_file = png_file.replace('.png', '.json')
        json_path = os.path.join(GRAPHICS_DIR, json_file)
        
        if os.path.exists(json_path):
            # Update the JSON file
            print(f"Updating {json_file}...")
            update_json(json_path, SCALE_FACTOR)
    
    print("Done! All PNG files have been upscaled and their JSON files updated.")

if __name__ == "__main__":
    main()