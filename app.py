
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI(title="Real-Time Object Detection API")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"

# Load the model once when the application starts.
model = YOLO(str(MODEL_PATH))


@app.get("/")
def home():
    # return {
    #     "message": "Object Detection API is running",
    #     "websocket": "/ws/detect"
    # }
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/detect")
async def detect_objects(websocket: WebSocket):
    await websocket.accept()
    print("Browser connected.")

    try:
        while True:
            # Receive one JPEG-encoded frame from the browser.
            image_bytes = await websocket.receive_bytes()

            image_array = np.frombuffer(
                image_bytes, dtype=np.uint8
            )
            frame = cv2.imdecode(
                image_array, cv2.IMREAD_COLOR
            )

            if frame is None:
                await websocket.send_json({
                    "error": "Invalid image frame"
                })
                continue

            # Run YOLO on this frame.
            results = model.predict(
                source=frame,
                conf=0.5,
                verbose=False
            )

            detections = []

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                detections.append({
                    "label": model.names[class_id],
                    "confidence": round(confidence, 3),
                    "box": [x1, y1, x2, y2]
                })

            # Return detection results to the browser.
            await websocket.send_json({
                "detections": detections,
                "width": frame.shape[1],
                "height": frame.shape[0]
            })

    except WebSocketDisconnect:
        print("Browser disconnected.")
