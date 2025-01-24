from builtins import enumerate
import streamlit as st
import cv2
import numpy as np
import math
import os
import subprocess

def classic_analys(tracking, x, y, w, h):
    str_name = "abcdefghijqlmnopqrstuvwxyz"
    min_dist = 1000000000000
    mini = ""
    
    for k in tracking.keys():
        p_x, p_y = tracking[k][len(tracking[k])-1][0], tracking[k][len(tracking[k])-1][1]
        
        dist = math.sqrt((p_x - x)**2 + (p_y - y)**2)
        
        if dist < min_dist:
            min_dist = dist
            mini = k
            
    if min_dist > 55:
        mini = str_name[len(tracking.keys())]
    
    return mini

def yolo_model():
    return 'oui'

def convert_video(input_path, output_path):
    """Convert video to web-compatible format using ffmpeg"""
    try:
        subprocess.run([
            'ffmpeg', 
            '-i', input_path, 
            '-codec:v', 'libx264', 
            '-preset', 'medium', 
            '-crf', '23', 
            '-codec:a', 'aac', 
            output_path
        ], check=True)
        return True
    except Exception as e:
        st.error(f"Video conversion error: {e}")
        return False

def process_video(video_path):
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128)
    ]

    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)
    if os.path.exists('outputs/web_video.mp4'):
        os.remove('outputs/web_video.mp4')
    output_path = 'outputs/processed_video.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    tracking = {}
    frame_count = 0

    progress_bar = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=60,
            param2=55,
            minRadius=300,
            maxRadius=370
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :2]:
                x, y, r = circle
                cv2.circle(frame, (x, y), 327, (0, 255, 0), 3)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        fgmask = fgbg.apply(gray)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ct_in_frame = len(contours)
        cv2.putText(frame, f"Daphnie: {ct_in_frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if ct_in_frame < 15:
            for (i, contour) in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                ll = classic_analys(tracking, x, y, w, h)

                if ll in tracking:
                    tracking[ll].append((x, y, w, h))
                else:
                    tracking[ll] = []
                    tracking[ll].append((x, y, w, h))

                cv2.putText(frame, ll, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for i, k in enumerate(tracking.keys()):
                points = tracking[k]
                if len(points) > 1:
                    for j, p in enumerate(points):
                        if j > 0:
                            cv2.line(frame, (points[j-1][0], points[j-1][1]), (points[j][0], points[j][1]), colors[i % len(colors)], 2)

        out.write(frame)

    cap.release()
    out.release()
    
    return output_path

def main():
    st.title('Daphnie Tracking Video Processor')

    dirname = 'videos/'
    video_name = st.radio("Choose a video", ["P1_24h_01top.mp4", "P1_48h_02mid.mp4", "P2_24h_01top.mp4"])

    genre = st.radio(
        "Choose Analysis Method",
        ["Analyse Classique", "Entrainement Supervise", "Entrainement Non-Supervise"],
        captions=[
            "OpenCv",
            "Detectron2, YOLO",
            "SAM, SAM2",
        ],
    )

    if genre == "Analyse Classique":
        
        # Process video
        st.video(dirname + video_name.split('.')[0] + "_ALGO.mp4")
        #st.video(dirname + ???)

    elif genre == "Entrainement Supervise":

        model_choice = st.radio(
            "Choose Non-Supervised Model",
            ["YOLO", "Detecron2"]
        )

        if model_choice == "YOLO":
            #result = yolo_model()
            st.video(dirname + video_name.split('.')[0] + "_YOLO.mp4")
            #st.video(dirname + video_name)
        
        elif model_choice == "Detecron2":
            st.video(dirname + video_name.split('.')[0] + "_detectron2.mp4")



    elif genre == "Entrainement Non-Supervise":

        model_choice = st.radio(
            "Choose Non-Supervised Model",
            ["SAM", "SAM + ALGO"]
        )
        st.write("Avec Algo est plus performante:")

        if model_choice == "SAM":
            st.video(dirname + video_name.split('.')[0] + "_Sam_nul.mp4")
        
        if model_choice == "SAM + ALGO":
            st.video(dirname + video_name.split('.')[0] + "_SAM_ALGO.mp4")



            
if __name__ == "__main__":
    main()