import cv2
import time
from ultralytics import YOLO
import os
import csv
import time

def process_video(video_path):
    max_current_total = 0

    os.makedirs("outputs", exist_ok=True) # Create the output foler...
    os.makedirs("logs", exist_ok=True) # Create the logs folder...
    
    VIDEO_PATH = video_path         
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    print("Video Resolution:", frame_width, "x", frame_height)
    print("Video FPS:", fps_video)


    csv_file = open(
    "logs/traffic_report.csv",
    mode="w",
    newline="",
    encoding="utf-8-sig")

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
    "Timestamp",
    "Frame",
    "Cars",
    "Buses",
    "Trucks",
    "Motorcycles",
    "Total Vehicles",
    "Density",
    "Traffic"])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
    "outputs/processed_traffic.mp4",
    fourcc,
    fps_video,
    (frame_width, frame_height))

    MODEL_PATH = "yolov8n.pt"

    DISPLAY_WIDTH = 1000
    DISPLAY_HEIGHT = 600

    CONFIDENCE_THRESHOLD = 0.5
    VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

    COLORS = {
    "car": (0, 255, 0),          # Green
    "bus": (255, 0, 0),          # Blue
    "truck": (0, 0, 255),        # Red
    "motorcycle": (0, 255, 255) }

    # Counting Line
    COUNT_LINE_Y = 450 
    OFFSET = 50 # Tolerance...

# ==================================================
# LOAD MODEL
# ==================================================

    print("Loading YOLOv8...")

    model = YOLO(MODEL_PATH)

    print("Model Loaded Successfully!")


    if not cap.isOpened():
        print("Error: Cannot open video!")
        exit()

    frame_number = 0
    previous_time = time.time()

# ==================================================
# VEHICLE COUNTERS
# ==================================================

    counted_ids = set()

    car_total = 0
    bus_total = 0
    truck_total = 0
    motorcycle_total = 0

# ==================================================
# MAIN LOOP
# ==================================================

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

    # Draw Counting Line
        cv2.line(
        frame,
        (0, COUNT_LINE_Y),
        (frame.shape[1], COUNT_LINE_Y),
        (0, 255, 255),
        3)

        cv2.putText(
        frame,
        "COUNTING LINE",
        (20, COUNT_LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2)

    # Tracking
        results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False)

    # Current Frame Counts
        car_count = 0
        bus_count = 0
        truck_count = 0
        motorcycle_count = 0

        if results and results[0].boxes.id is not None:

            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, ids):

                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                cls = int(box.cls[0])
                label = model.names[cls]
                print(f"Detected: {label} | ID: {track_id} | Confidence: {confidence:.2f}")

                if label not in VEHICLE_CLASSES:
                    continue

            # Current Counts
                if label == "car":
                    car_count += 1
                elif label == "bus":
                    bus_count += 1
                elif label == "truck":
                    truck_count += 1
                elif label == "motorcycle":
                    motorcycle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center Point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                color = COLORS[label]
    
                # Draw Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2)

            # Draw Center
                cv2.circle(
                    frame, (cx, cy), 6, 
                    (0, 0, 255), -1)

            # Draw Label
                cv2.putText(
                    frame,
                    f"ID:{track_id} {label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

            # Count each unique tracked vehicle only once
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
    
                    if label == "car":
                        car_total += 1

                    elif label == "bus":
                        bus_total += 1
    
                    elif label == "truck":
                        truck_total += 1
    
                    elif label == "motorcycle":
                        motorcycle_total += 1

    # ==================================================
    # TOTALS
    # ==================================================

        current_total = (
            car_count +
            bus_count +
            truck_count +
            motorcycle_count
        )

            # Maximum vehicles expected in one frame

        MAX_VEHICLES = 15
    
        # Alert threshold
        ALERT_THRESHOLD = 80  # percent

        density = min((current_total / MAX_VEHICLES) * 100, 100)

    # Save statistics every 30 frames
        if frame_number % 30 == 0:
            seconds = int(frame_number / fps_video)
    
            timestamp = time.strftime(
            "%H:%M:%S",
            time.gmtime(seconds))


        total_crossed = (
            car_total +
            bus_total +
            truck_total +
            motorcycle_total
        )

    # Traffic Density
        if density < 30:
            traffic = "LIGHT"
            traffic_color = (0, 255, 0)

        elif density < 70:
            traffic = "MODERATE"
            traffic_color = (0, 255, 255)
        
        else:
            traffic = "HEAVY"
            traffic_color = (0, 0, 255)
        if frame_number % 30 == 0:

            seconds = int(frame_number / fps_video)
        
            timestamp = time.strftime(
                "%H:%M:%S",
                time.gmtime(seconds)
            )
            csv_writer.writerow([
        timestamp,
        frame_number,
        car_count,
        bus_count,
        truck_count,
        motorcycle_count,
        current_total,
        round(density,1),
        traffic])


    
        # ==================================================
        # DASHBOARD
        # ==================================================
    
        cv2.rectangle(
        frame,
        (10, 10),
        (370, 420),
        (40, 40, 40),
        -1)

        y = 35
    
        cv2.putText(frame, "AI SMART TRAFFIC MONITOR",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)
    
        y += 40
        

    
        cv2.putText(frame,f"Cars : {car_count}",(20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,COLORS["car"],2)
    
        y += 30
    
        cv2.putText(frame,f"Buses : {bus_count}",(20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,COLORS["bus"],2)
    
        y += 30
    
        cv2.putText(frame,f"Trucks : {truck_count}",(20,y),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,COLORS["truck"],2)

        y += 30
    
        cv2.putText(frame,f"Motorcycles : {motorcycle_count}",(20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,COLORS["motorcycle"],2)
    
        y += 40
    
        cv2.putText(frame,f"Current Vehicles : {current_total}",
                    (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255,255,255),
                    2)
    
        y += 35

        cv2.putText(frame,f"Vehicles Crossed : {total_crossed}",
                    (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0,255,255),
                    2)
    
        y += 35

        cv2.putText(frame,
    f"Density : {density:.1f}%",
    (20, y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255,255,255),
    2)

        y += 35
    
        cv2.putText(
    frame,
    f"Traffic : {traffic}",
    (20,y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    traffic_color,
    2)
        
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
    
        # FPS
        cv2.putText(frame,
                    f"FPS : {fps:.1f}",
                    (760,35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)
    
        # Frame
        cv2.putText(frame,
                    f"Frame : {frame_number}",
                    (760,70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)
    
        cv2.putText(frame,
                "Press Q to Exit",
                (720,105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200,200,200),
                2)

    # Resize Window
        display = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )
        # Save original processed frame
        out.write(frame)
    
        cv2.imshow(
            "AI Smart Traffic Monitoring",
            display
        )
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ==================================================
# SUMMARY
# ==================================================

    print("\n========== SUMMARY ==========")
    print(f"Frames Processed : {frame_number}")
    print(f"Cars Counted     : {car_total}")
    print(f"Buses Counted    : {bus_total}")
    print(f"Trucks Counted   : {truck_total}")
    print(f"Bikes Counted    : {motorcycle_total}")
    print(f"Total Vehicles   : {total_crossed}")
    
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print("Returned values:")
    print({
    "cars": car_total,
    "density": density,
    "traffic": traffic})

    return {

    "cars": car_total,
    "buses": bus_total,
    "trucks": truck_total,
    "motorcycles": motorcycle_total,

    "total": total_crossed,

    "density": round(density,1),

    "traffic": traffic,

    "processed_video":"outputs/processed_traffic.mp4",

    "csv_report":"logs/traffic_report.csv"}

if __name__ == "__main__":
    process_video("assets/131232-749706873.mp4")
           