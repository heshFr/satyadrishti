"""
Satya Drishti
=============
Downloads the FaceForensics++ dataset using the officially provided
Python scripts. Requires an agreement form to be submitted to the 
authors first to acquire a download link.

This wrapper automates face extraction via MediaPipe post-download.
"""

import os
import argparse
import subprocess
import cv2

try:
    import mediapipe as mp
    HAS_MP = True
    mp_face_detection = mp.solutions.face_detection
except ImportError:
    HAS_MP = False

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "faceforensics")

def clone_ff_downloader():
    """Clones the official FF++ repository for the downloader script."""
    repo_url = "https://github.com/ondyari/FaceForensics.git"
    target_dir = os.path.join(DATA_DIR, "repo")
    
    if not os.path.exists(target_dir):
        print(f"Cloning official FaceForensics repository to {target_dir}...")
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
    return os.path.join(target_dir, "dataset", "download-FaceForensics.py")

def download_dataset(script_path, output_dir, dataset_type="manipulated_sequences", link=""):
    """Runs the official FF++ download script."""
    if not link:
         print("Error: FaceForensics++ requires a secure download link provided by the authors.")
         print("Please visit: https://github.com/ondyari/FaceForensics and sign the agreement.")
         return False
         
    cmd = [
        "python", script_path,
        output_dir,
        "-d", dataset_type,
        "-c", "c23", # Compression level: RAW, c23 (high quality), c40 (low quality)
        "--link", link
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    return True

def extract_faces(video_dir, output_dir, target_size=(224, 224)):
    """Extracts faces from downloaded FF++ videos using MediaPipe."""
    if not HAS_MP:
        print("MediaPipe not installed. Skipping face extraction.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    
    print(f"Extracting faces from {len(video_files)} videos...")
    for video_name in video_files:
        video_path = os.path.join(video_dir, video_name)
        base_name = os.path.splitext(video_name)[0]
        vid_out_dir = os.path.join(output_dir, base_name)
        os.makedirs(vid_out_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame to save space
            if frame_idx % 5 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb_frame)
                
                if results.detections:
                    # Take highest confidence face
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                                 
                    # Add 20% margin
                    margin_x, margin_y = int(w*0.2), int(h*0.2)
                    x = max(0, x - margin_x)
                    y = max(0, y - margin_y)
                    w = min(iw - x, w + 2*margin_x)
                    h = min(ih - y, h + 2*margin_y)
                    
                    face_crop = frame[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        face_resized = cv2.resize(face_crop, target_size)
                        out_file = os.path.join(vid_out_dir, f"frame_{frame_idx:04d}.jpg")
                        cv2.imwrite(out_file, face_resized)
                        
            frame_idx += 1
        cap.release()
    detector.close()
    print("Face extraction complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", type=str, help="Secure download link from FF++ authors", default="")
    parser.add_argument("--extract-only", action="store_true", help="Skip download, just run face extraction")
    args = parser.parse_args()
    
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_dir = os.path.join(DATA_DIR, "raw_videos")
    faces_dir = os.path.join(DATA_DIR, "faces")
    
    if not args.extract_only:
        script_path = clone_ff_downloader()
        success = download_dataset(script_path, raw_dir, link=args.link)
        if not success:
             return
             
    if os.path.exists(raw_dir):
        extract_faces(raw_dir, faces_dir)
    else:
        print(f"Raw video directory {raw_dir} not found. Cannot extract faces.")

if __name__ == "__main__":
    main()
