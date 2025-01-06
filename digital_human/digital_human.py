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


def main(video_path, text):
    audio_file = generate_audio(text)
    print(f"Audio file saved as: {audio_file}")
    output_video = run_wav2lip(video_path, audio_file)
    print(f"Lip-synced video saved as: {output_video}")
    if os.path.exists(audio_file):
        os.remove(audio_file)

    print(f"Output video saved as: {output_video}")


# if __name__ == "__main__":
#     video_input_path = "input_video.mp4"  # Path to your input video file
#     text_input = "Hello! I am speaking this text from the input. This is a test. We are a team of five people. We are learning how to code."  # Text to be spoken
#
#     main(video_input_path, text_input)

# --checkpoint_path checkpoints/wav2lip_gan.pth --face input_video.mp4  --audio content.wav --outfile output_video.mp4


# Multi threading

import argparse
import concurrent.futures
import time

import concurrent.futures
import time  # Simulate processing time


def mainMain():
    max_threads = 10
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        futures = []

        while True:
            try:
                user_input = input("Enter video_path, audio_path, and text (comma-separated): ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting program. Waiting for ongoing tasks to complete...")
                    break

                video_path, audio_path, text = map(str.strip, user_input.split(","))
            except ValueError:
                print("Invalid input. Please provide video_path, audio_path, and text as comma-separated values.")
                continue

            # Submit the task to a thread
            future = executor.submit(main, video_path, text)
            futures.append(future)

            # Clean up completed futures
            futures = [f for f in futures if not f.done()]

            # Limit the number of threads
            while len(futures) >= max_threads:
                concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                futures = [f for f in futures if not f.done()]


if __name__ == "__main__":
    mainMain()
