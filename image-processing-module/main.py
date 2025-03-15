import cv2
import numpy as np
import os
import random
import tempfile
import io
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### --- Utility functions ---

def crop_to_aspect_ratio(crop_img: np.ndarray, target_aspect: float):
    """
    Crop the image to the target aspect ratio (width/height) without stretching.
    """
    h, w = crop_img.shape[:2]
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_w = int(target_aspect * h)
        offset = (w - new_w) // 2
        cropped = crop_img[:, offset:offset+new_w]
    elif current_aspect < target_aspect:
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        cropped = crop_img[offset:offset+new_h, :]
    else:
        cropped = crop_img
    return cropped

def create_crop_with_face_centered(image: np.ndarray, face_coords: tuple) -> np.ndarray:
    """
    Given an image and face coordinates (x, y, w, h), create a crop where:
      - The crop height is two times the face height (adjustable).
      - The crop width is calculated to preserve the original image's aspect ratio.
      - The face is centered in the crop.
    The crop boundaries are adjusted to remain within the image.
    """
    face_x, face_y, face_w, face_h = face_coords
    
    # Calculate face center.
    face_center_x = face_x + face_w / 2
    face_center_y = face_y + face_h / 2

    # Desired crop height: two times the face height.
    crop_h = 2 * face_h
    
    # Use original image's aspect ratio.
    orig_aspect = image.shape[1] / image.shape[0]
    crop_w = crop_h * orig_aspect

    # Compute top-left coordinate such that the face center is centered in the crop.
    crop_x = int(face_center_x - crop_w / 2)
    crop_y = int(face_center_y - crop_h / 2)
    
    # Adjust boundaries.
    if crop_x < 0:
        crop_x = 0
    if crop_y < 0:
        crop_y = 0
    if crop_x + crop_w > image.shape[1]:
        crop_x = image.shape[1] - int(crop_w)
    if crop_y + crop_h > image.shape[0]:
        crop_y = image.shape[0] - int(crop_h)

    crop_w = int(crop_w)
    crop_h = int(crop_h)
    crop = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    h, w = image.shape[:2]
    scale = min(640 / w, 640 / h)
    new_width = int(w * scale)
    new_height = int(h * scale)

    res = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return res

### --- Video frame extraction ---

def extract_random_frames_from_video(video_path, num_frames=60, segment_minutes=10):
    """
    Extracts up to num_frames random frames from the first `segment_minutes` of the video.
    Returns a list of tuples (frame_index, frame) for frames extracted in memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_frame_count = int(fps * segment_minutes * 60)
    segment_frame_count = min(segment_frame_count, total_frames)
    
    if segment_frame_count <= num_frames:
        sample_indices = list(range(segment_frame_count))
    else:
        sample_indices = sorted(random.sample(range(segment_frame_count), num_frames))
    
    frames = []
    current_frame_index = 0
    while current_frame_index < segment_frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame_index in sample_indices:
            frames.append((current_frame_index, frame))
        current_frame_index += 1
    cap.release()
    return frames

### --- DeepFace analysis for a single frame ---

def analyze_frame(frame: np.ndarray, sentiment: str) -> np.ndarray:
    """
    Encodes the given frame, calls the DeepFace API to analyze emotions,
    and if a face with a matching sentiment (by dominant emotion and highest score)
    is found, returns the cropped frame (using create_crop_with_face_centered).
    Otherwise, returns None.
    """
    # Encode the frame to JPEG.
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return None
    image_bytes = buffer.tobytes()
    
    # DeepFace API endpoint.
    analyze_url = "http://deepface:5000/analyze"
    files = {"img": ("frame.jpg", image_bytes, "image/jpeg")}
    data = {"actions": "emotion"}
    
    try:
        r = httpx.post(analyze_url, files=files, data=data, timeout=20.0)
        r.raise_for_status()
    except Exception as e:
        print(f"DeepFace analyze error on frame: {e}")
        return None
    
    analysis_data = r.json()
    faces = analysis_data.get("results", [])
    if not faces:
        return None
    
    matching_face = None
    highest_score = -99999
    # Set which sentiments to consider.
    sentiments = ["angry", "disgust", "fear", "sad", "neutral", "happy", "surprise"]
    if sentiment.lower() == "negative":
        sentiments = ["angry", "disgust", "fear", "sad"]
    elif sentiment.lower() == "positive":
        sentiments = ["happy", "surprise"]
    elif sentiment.lower() == "neutral":
        sentiments = ["neutral"]
    
    for face in faces:
        for _sent in sentiments:
            if face.get("dominant_emotion", "").lower() == _sent.lower():
                print("Dominant: ", face.get("dominant_emotion"))
                region = face.get("region", {})
                if all(k in region for k in ("x", "y", "w", "h")):
                    emotion_scores = face.get("emotion", {})
                    score = emotion_scores.get(_sent.lower(), 0)
                    if score > highest_score:
                        highest_score = score
                        matching_face = (region["x"], region["y"], region["w"], region["h"])
    
    if matching_face:
        res = dict()
        res["score"] = highest_score
        res["image"] = create_crop_with_face_centered(frame, matching_face)
        return res
    return None

### --- Process video frames in parallel --- 

def process_video_file(video_path: str, sentiment: str) -> np.ndarray:
    """
    Extracts random frames from the video and analyzes them in parallel (10 at a time)
    using the DeepFace API. Returns the first processed frame (cropped) that has a face
    matching the given sentiment. If none of the frames yield a matching face, returns None.
    """
    frames = extract_random_frames_from_video(video_path)
    if not frames:
        return None

    result_img = None
    highest_score = -99999
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit analysis tasks for each frame.
        futures = {executor.submit(analyze_frame, frame, sentiment): idx for idx, frame in frames}
        for future in futures:
            try:
                res = future.result()
                if res is not None and res["score"] > highest_score:
                    # Return first frame with a valid matching face.
                    result_img = res["image"]
                    highest_score = res["score"]
                    break
            except Exception as e:
                print(f"Error processing frame: {e}")
    if result_img is None:
        # Cycle through all emotions, and store first available face
        for _sent in ["happy", "surprise", "neutral", "sad", "angry", "disgust", "fear"]:
            with ThreadPoolExecutor(max_workers=8) as executor:
              # Submit analysis tasks for each frame.
              futures = {executor.submit(analyze_frame, frame, _sent): idx for idx, frame in frames}
              for future in futures:
                  try:
                      res = future.result()
                      print("Edge case res: ", res)
                      if res is not None and res["image"] is not None:
                          # fallback to any face
                          result_img = res["image"]
                          break
                  except Exception as e:
                      print(f"Error processing frame: {e}")
            if result_img is not None:
                break
    return result_img

### --- Endpoint: Process Video Input ---

@app.post("/process_video")
async def process_video_endpoint(
    video: UploadFile = File(...),
    sentiment: str = Form(...)
):
    """
    Endpoint that accepts a video file and a sentiment string.
    It extracts random frames from the first 10 minutes of the video,
    analyzes up to 10 frames in parallel using the DeepFace API,
    and returns the processed (cropped) frame as a PNG image.
    """
    # Determine the file extension based on the uploaded file's name.
    filename_lower = video.filename.lower()
    if filename_lower.endswith(".webm"):
        suffix = ".webm"
    elif filename_lower.endswith(".mp4"):
        suffix = ".mp4"
    else:
        # Default to MP4 if extension is not recognized.
        suffix = ".mp4"

    # Save the uploaded video to a temporary file.
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await video.read())
            video_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving video: {e}")
    
    loop = asyncio.get_running_loop()
    result_img = await loop.run_in_executor(None, process_video_file, video_path, sentiment)
    os.remove(video_path)
    
    if result_img is None:
        raise HTTPException(status_code=404, detail="No frame with matching sentiment found.")
    
    ret, buf = cv2.imencode(".png", result_img)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
