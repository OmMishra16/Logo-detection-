import av
import os
import argparse
import json
import math
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Load the model
model = YOLO('model/best.pt')

# Labels mapping
labels = {0: "Pepsi", 1: "CocaCola"}

# Function to extract frames, detect logos, and annotate frames
def process_video(video_path, output_video_path):
    container = av.open(video_path)
    output_container = av.open(output_video_path, mode='w')

    stream = container.streams.video[0]
    codec_name = stream.codec_context.name
    stream_options = {
        'time_base': stream.time_base,
        'framerate': stream.average_rate,
        'pix_fmt': 'yuv420p',
        'height': stream.codec_context.height,
        'width': stream.codec_context.width
    }
    output_stream = output_container.add_stream(codec_name, stream.average_rate)
    output_stream.width = stream.codec_context.width
    output_stream.height = stream.codec_context.height
    output_stream.pix_fmt = 'yuv420p'

    detections = {"Pepsi_pts": [], "CocaCola_pts": []}

    for frame in container.decode(video=0):
        img = frame.to_image()
        timestamp = round(frame.time, 1)

        # Convert PIL Image to OpenCV format
        frame_data = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = model(frame_data)

        img_width, img_height = img.size

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = labels.get(cls_id, "Unknown")

                if box.xywh.shape[1] == 4:
                    x_center, y_center, width, height = box.xywh[0]
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    size = width * height
                    distance_from_center = math.sqrt((x_center - img_width / 2) ** 2 + (y_center - img_height / 2) ** 2)
                    detection_info = {
                        "timestamp": timestamp,
                        "size": size.item(),
                        "distance_from_center": distance_from_center
                    }
                    detections[f"{label}_pts"].append(detection_info)

                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                    # Draw label
                    draw.text((x1, y1 - 10), f'{label}', fill='red', font=font)

        # Convert annotated PIL Image back to OpenCV format for saving
        annotated_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Save frame to output video
        frame_out = av.VideoFrame.from_ndarray(annotated_frame, format='bgr24')
        for packet in output_stream.encode(frame_out):
            output_container.mux(packet)

    # Flush and close the container
    for packet in output_stream.encode():
        output_container.mux(packet)
    output_container.close()

    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from video, detect logos, and save annotated video')
    parser.add_argument('--video_path', type=str, required=True, help='/Users/ommishra/Desktop/machineLearningAssignment/demo_video.mp4')
    parser.add_argument('--output_file', type=str, required=True, help='/Users/ommishra/Desktop/machineLearningAssignment/detections.json')
    parser.add_argument('--output_video_path', type=str, required=True, help='/Users/ommishra/Desktop/machineLearningAssignment')

    args = parser.parse_args()
    detections = process_video(args.video_path, args.output_video_path)

    with open(args.output_file, 'w') as json_file:
        json.dump(detections, json_file, indent=4)

    print(f"Detection results saved to {args.output_file}")
    print(f"Annotated video saved to {args.output_video_path}")
