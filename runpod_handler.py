import sys
import subprocess
import runpod
import os
import tempfile
import requests
import uuid
import base64, json
from google.cloud import storage
from predict import Predictor # Assuming Predictor is in predict.py

# Global predictor instance to reuse (RunPod might keep workers warm)
# Setup will be called if the instance is not ready
predictor_instance = None

# ------------------ GCS client helper ------------------
_gcs_client = None


def get_gcs_client():
    """Initialize (or return cached) Google Cloud Storage client."""
    global _gcs_client
    if _gcs_client is not None:
        return _gcs_client

    sa_b64 = os.getenv("GCS_SA_BASE64")
    if not sa_b64:
        raise RuntimeError("GCS_SA_BASE64 environment variable not set. Cannot init GCS client.")

    key_path = "/tmp/gcs_sa.json"
    if not os.path.exists(key_path):
        with open(key_path, "wb") as f:
            f.write(base64.b64decode(sa_b64))

    _gcs_client = storage.Client.from_service_account_json(key_path)
    return _gcs_client
# -------------------------------------------------------


def download_file(url, destination_dir):
    """Downloads a file from a URL to a specified directory using pget."""
    print(f"Downloading {url} to {destination_dir}")
    
    # Extract a clean filename from URL (remove query parameters)
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    clean_filename = os.path.basename(parsed_url.path)
    if not clean_filename or clean_filename == '/':
        clean_filename = "downloaded_file"
    
    filename = os.path.join(destination_dir, clean_filename)
    
    try:
        subprocess.check_call(["pget", url, filename], close_fds=False)
        print(f"Successfully downloaded {url} to {filename}")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        raise
    except FileNotFoundError:
        print("Error: pget command not found. Ensure it is installed and in PATH.")
        raise Exception("pget not found. Cannot download files.")


def handler(event):
    global predictor_instance

    job_input = event.get('input', {})
    if not job_input:
        return {"error": "No input provided in event"}

    video_url = job_input.get('video_url')
    audio_url = job_input.get('audio_url')
    audio_max_sec = float(job_input.get('audio_max_sec', 0))  # 0 means no clipping

    if not video_url or not audio_url:
        return {"error": "video_url and audio_url are required."}

# Parameters for the model, with defaults matching predict.py if not provided
    guidance_scale = float(job_input.get('guidance_scale', 2.0))
    inference_steps = int(job_input.get('inference_steps', 20))
    seed = int(job_input.get('seed', 0)) # 0 for random seed as per predict.py
    remove_background = bool(job_input.get('remove_background', False))

    # Initialize predictor if not already done
    if predictor_instance is None:
        print("Initializing Predictor...")
        predictor_instance = Predictor()
        predictor_instance.setup() # This downloads model weights, sets up links
        print("Predictor initialized.")

    try:
        # Create temporary directories for downloaded files
        with tempfile.TemporaryDirectory() as tmpdir_video, tempfile.TemporaryDirectory() as tmpdir_audio:
            print(f"Created temporary directories: {tmpdir_video}, {tmpdir_audio}")
            
            local_video_path = download_file(video_url, tmpdir_video)
            local_audio_path = download_file(audio_url, tmpdir_audio)

            # Optional audio clipping
            if audio_max_sec > 0:
                clipped_audio = os.path.join(tmpdir_audio, "clip.wav")
                try:
                    subprocess.check_call([
                        "ffmpeg", "-y", "-i", local_audio_path,
                        "-t", str(audio_max_sec), "-c", "copy", clipped_audio
                    ])
                    local_audio_path = clipped_audio
                    print(f"Audio clipped to {audio_max_sec}s -> {clipped_audio}")
                except subprocess.CalledProcessError as e:
                    print(f"Error clipping audio: {e}")
                    return {"error": "Failed to clip audio"}

            print(f"Calling predictor.predict with video: {local_video_path}, audio: {local_audio_path}")
            # Call the predict method
            output_path_object = predictor_instance.predict(
                video=local_video_path, # predict.py expects string paths
                audio=local_audio_path, # predict.py expects string paths
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                seed=seed,
                remove_background=remove_background
            )
            
            output_path_str = str(output_path_object) # Convert Path object to string
            print(f"Prediction successful. Output at: {output_path_str}")

            # ---------- Upload to Google Cloud Storage ----------
            local_file_path = output_path_str  # '/tmp/video_out.mp4'

            bucket_name = os.getenv("GCS_BUCKET")
            if not bucket_name:
                print("HANDLER: Error - GCS_BUCKET environment variable not set.")
                return {"error": "GCS_BUCKET environment variable not set."}

            try:
                client = get_gcs_client()
                bucket = client.bucket(bucket_name)

                prefix = os.getenv("GCS_UPLOAD_PREFIX", "videos/")
# MODNet使用MP4格式，文件更小
                file_extension = "mp4"
                content_type = "video/mp4"
                unique_filename = f"{uuid.uuid4()}.{file_extension}"
                blob_path = f"{prefix}{unique_filename}"
                blob = bucket.blob(blob_path)

                print(f"HANDLER: Uploading to gs://{bucket_name}/{blob_path} ...")
                blob.upload_from_filename(local_file_path, content_type=content_type)

                public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_path}"
                print(f"HANDLER: Upload success. Public URL: {public_url}")

                return {"output_url": public_url}
            except Exception as upload_e:
                print(f"HANDLER: Failed to upload to GCS: {upload_e}")
                return {"error": "Failed to upload to GCS", "details": str(upload_e)}
            # ---------- End GCS upload ----------

    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"HANDLER: Exception caught ({e}), returning error.")
        return {"error": str(e)}

if __name__ == '__main__':
    # This part is for local testing of the handler, if needed.
    # It simulates a RunPod event.
    # You would need to have pget installed and model weights accessible.
    # And ensure predict.py and its dependencies are in PYTHONPATH.
    
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
