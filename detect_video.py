
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
VIDEO_PATH = BASE_DIR / "videos" / "15436787_1080_1920_30fps.mp4"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "detected_video.mp4"

CONFIDENCE_THRESHOLD = 0.5
MODEL_NAME = "yolo11n.pt"


def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        print("Place an MP4 video at videos/15436787_1080_1920_30fps.mp4")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pretrained YOLO model
    model = YOLO(MODEL_NAME)

    # Open the source video
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open the video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        print("Invalid video properties.")
        cap.release()
        return

    # MP4 video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("Could not create output video.")
        cap.release()
        return

    frame_number = 0

    window_name = "Phase 2 - Video Object Detection"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 720, 1280)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Run YOLO on the current frame
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()

            # Write annotated frame to output video
            writer.write(annotated_frame)

            # Display the processed video
            # cv2.imshow("Phase 2 - Video Object Detection", annotated_frame)
            cv2.imshow(window_name, annotated_frame)

            # Print progress periodically
            if frame_number % 30 == 0:
                print(
                    f"Processed {frame_number}/{total_frames} frames"
                )

            # Press Q to stop early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Processed frames: {frame_number}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
