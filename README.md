# Logo Detection in Videos

This project uses a YOLOv8 model to detect Pepsi and CocaCola logos in video files. The output includes both a JSON file with detection details and an annotated video showing the detections.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Output Format](#output-format)
- [Demo Video](#demo-video)


## Installation

### Prerequisites

- Python 3.7 or higher
- FFmpeg and libav installed on your system (for handling video files)

### Installing Dependencies

1. **Clone the repository:**

    ```bash
    git clone <repo_url>
    cd logo_detection_pipeline
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv logo_detection_env
    source logo_detection_env/bin/activate  # On Windows use: logo_detection_env\Scripts\activate
    ```

3. **Install required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install FFmpeg and libav (if not already installed):**

    - On macOS:

        ```bash
        brew install ffmpeg
        brew install libav
        ```

    - On Ubuntu:

        ```bash
        sudo apt update
        sudo apt install ffmpeg libavcodec-extra
        ```

## Usage

1. **Ensure the video file (`demo_video.mp4`) and the trained model file (`best.pt`) are in the correct locations:**
    - Place your input video file in the project directory or update the path accordingly in the script arguments.
    - Ensure `best.pt` is in the `model` directory.

2. **Example Command to Run the script from the terminal:**

    ```bash
    python3 logo_detection.py --video_path /Users/ommishra/Desktop/machineLearningAssignment/demo_video.mp4 --output_file /Users/ommishra/Desktop/machineLearningAssignment/detections.json --output_video_path /Users/ommishra/Desktop/machineLearningAssignment/annotated_video.mp4
    ```

### Example Command:

```bash
python3 logo_detection.py --video_path /path/to/your/demo_video.mp4 --output_file /path/to/your/detections.json --output_video_path /path/to/your/annotated_video.mp4

```

### Directory Structure
```
logo_detection_pipeline/
├── README.md
├── requirements.txt
├── logo_detection.py
├── model/
│   └── best.pt
├── demo_video.mp4
└── detections.json
```
### Output Format 

```
{
    "Pepsi_pts": [
        {
            "timestamp": 7.6,
            "size": 11336.6494140625,
            "distance_from_center": 347.8847353111372
        }
],
"CocaCola_pts": [
        {
            "timestamp": 1.2,
            "size": 245788.453125,
            "distance_from_center": 58.005440292403826
        }
]
}
```
