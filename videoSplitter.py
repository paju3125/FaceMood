from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(video_path, output_video_path, output_audio_path):
    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Extract video without audio
    video_clip.write_videofile(output_video_path, codec="libx264", audio=False)

    # Extract audio only
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)

    # Close the clips
    video_clip.close()
    audio_clip.close()

if __name__ == "__main__":
    input_video_path = 'VEGEMITE62_2.mp4'  # Replace with the path to your input video
    output_video_path = 'video splitter output/output_video.mp4'  # Replace with the desired path for the output video
    output_audio_path = 'video splitter output/output_audio.wav'  # Replace with the desired path for the output audio

    split_video(input_video_path, output_video_path, output_audio_path)
