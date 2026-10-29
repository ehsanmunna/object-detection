
from pathlib import Path
from ultralytics import YOLO

# Project directories
BASE_DIR = Path(__file__).parent
IMAGE_PATH = BASE_DIR / "images" / "dataset-1.png"
OUTPUT_DIR = BASE_DIR / "output"

def main():
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}")
        print("Put an image at images/dataset-1.png")
        return

    # Load a pretrained, lightweight object detection model
    model = YOLO("yolo11n.pt")

    # Detect objects in the image
    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.5,
        save=False
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    for result in results:
        # Draw bounding boxes and labels
        annotated_image = result.plot()

        output_path = OUTPUT_DIR / "detected.jpg"

        import cv2
        cv2.imwrite(str(output_path), annotated_image)

        print("\nDetected objects:")

        for box in result.boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]
            confidence = float(box.conf[0])

            print(f"{name}: {confidence:.2%}")

        print(f"\nOutput saved to: {output_path}")

        # Display the result
        import cv2
        cv2.imshow("Image Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
