from moviepy.editor import VideoFileClip

def reduce_resolution_to_240p(input_path, output_path):
    """
    Reduces the resolution of a video to 240p.
    
    Parameters:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
    """
    try:
        # Load the video file
        clip = VideoFileClip(input_path)
        
        # Set height to 240 pixels while maintaining the aspect ratio
        new_height = 240
        aspect_ratio = clip.w / clip.h
        new_width = int(aspect_ratio * new_height)
        
        # Resize the video
        resized_clip = clip.resize(height=new_height, width=new_width)
        
        # Write the resized video to output file
        resized_clip.write_videofile(output_path, codec="libx264")
        
        print(f"Video successfully resized to 240p and saved as {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_video = "../vids/hd/v1.mp4"  # Replace with your input file path
output_video = "../vids/and/v1and.mp4"  # Replace with your desired output file path
reduce_resolution_to_240p(input_video, output_video)