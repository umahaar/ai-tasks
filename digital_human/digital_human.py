import gradio as gr
import subprocess
import os
from gtts import gTTS

def generate_audio(text, filename="output.wav"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

def run_wav2lip(video_path, audio_path):
    output_video_path = "output_video.mp4"

    command = [
        "python", "inference.py",
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_video_path
    ]

    subprocess.run(command, check=True)

    return output_video_path

def wav2lip_pipeline(video, text, audio=None):
    try:
        # Use video file path directly
        video_path = video

        # Check if audio is provided; if not, generate it
        if audio is not None:
            audio_path = audio  # `audio` is a file path
        else:
            audio_path = generate_audio(text)

        # Run Wav2Lip
        output_video = run_wav2lip(video_path, audio_path)

        return output_video  # Return output video path

    except Exception as e:
        return f"Error: {str(e)}"

def gradio_interface(video, text, audio=None):
    result = wav2lip_pipeline(video, text, audio)
    if isinstance(result, str) and result.startswith("Error"):
        return result  # Return error message
    return result  # Return output video file path

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Input Video"),  # Removed `type="filepath"`
        gr.Textbox(label="Text to Sync"),
        gr.Audio(label="Optional Audio File", type="filepath")  # Ensure `type="filepath"`
    ],
    outputs=gr.Video(label="Lip-Synced Output Video"),  # No `type="filepath"` needed
    title="Wav2Lip Gradio Interface",
    description="Upload a video, provide text, and optionally upload an audio file. The model will sync the lips to the text and audio."
)

if __name__ == "__main__":
    interface.launch(share=True)
