
import os
import re


# Function to create a video from images in a folder
def create_video_from_images(folder_path, output_path, video_length=10):
    # Get list of image files sorted by name
    images = sorted(
        [img for img in os.listdir(folder_path) if re.match(r'\d+.png', img)],
        key=lambda x: int(re.match(r'(\d+).png', x).group(1))
    )
    if not images:
        raise ValueError('No images found in the folder.')

    # Calculate framerate based on the number of images and desired video length
    framerate = max(len(images) / video_length, 1)

    # Construct ffmpeg command
    ffmpeg_cmd = f"ffmpeg -framerate {framerate} -pattern_type glob -i '{folder_path}/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p {output_path}"
    os.system(ffmpeg_cmd)

    return f'Video created at {output_path}'


if __name__ == "__main__":
    create_video_from_images("/Users/nielswarncke/Documents/code/TransformerLens/demos/automated-interpretability/images/Reconstruction of each individual sparse feature (Identity is ideal)", "vid.mp4")