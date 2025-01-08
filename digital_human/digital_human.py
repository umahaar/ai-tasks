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
        # Save video file locally
        video_path = "temp_input_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video.read())

        # Use provided audio or generate one
        if audio:
            audio_path = "temp_input_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio.read())
        else:
            audio_path = generate_audio(text)

        # Run Wav2Lip
        output_video = run_wav2lip(video_path, audio_path)

        # Clean up temporary files
        os.remove(video_path)
        if audio is None:
            os.remove(audio_path)

        return output_video

    except Exception as e:
        return f"Error: {str(e)}"

def gradio_interface(video, text, audio=None):
    result = wav2lip_pipeline(video, text, audio)
    if isinstance(result, str) and result.startswith("Error"):
        return result  # Return error message
    return result  # Return output video

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Textbox(label="Text to Sync"),
        gr.Audio(label="Optional Audio File", optional=True)
    ],
    outputs=gr.Video(label="Lip-Synced Output Video"),
    title="Wav2Lip Gradio Interface",
    description="Upload a video, provide text, and optionally upload an audio file. The model will sync the lips to the text and audio."
)

# For Colab, use share=True to generate a public URL
interface.launch(share=True)
