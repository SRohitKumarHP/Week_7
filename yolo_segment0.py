import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# Define and parse user input arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    help='Path to YOLO segmentation model file (example: "runs/segment/train/weights/best.pt")',
    required=True
)

parser.add_argument(
    '--source',
    help='Image source, can be image file ("test.jpg"), image folder ("test_dir"), '
         'video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")',
    required=True
)

parser.add_argument(
    '--thresh',
    help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
    default=0.5
)

parser.add_argument(
    '--resolution',
    help='Resolution in WxH to process/display the frame at (example: "640x480"), '
         'otherwise, match source resolution',
    default=None
)

parser.add_argument(
    '--display',
    help='Initial output window size in WxH (example: "1280x720"). '
         'This only controls the display window size.',
    default='1280x720'
)

parser.add_argument(
    '--record',
    help='Record results from video or webcam and save it as "demo1.avi". '
         'Must specify --resolution argument to record.',
    action='store_true'
)

args = parser.parse_args()


# ============================================================
# Parse user inputs
# ============================================================

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
display_res = args.display
record = args.record


# ============================================================
# Check if model file exists and is valid
# ============================================================

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    print('Make sure the model filename was entered correctly.')
    sys.exit(0)


# ============================================================
# Load the YOLOv11s segmentation model
# ============================================================

model = YOLO(model_path, task='segment')
labels = model.names


# ============================================================
# Parse input to determine if source is image, folder,
# video, USB camera, or Picamera
# ============================================================

img_ext_list = [
    '.jpg', '.JPG',
    '.jpeg', '.JPEG',
    '.png', '.PNG',
    '.bmp', '.BMP'
]

vid_ext_list = [
    '.avi',
    '.mov',
    '.mp4',
    '.mkv',
    '.wmv'
]


if os.path.isdir(img_source):

    source_type = 'folder'

elif os.path.isfile(img_source):

    _, ext = os.path.splitext(img_source)

    if ext in img_ext_list:
        source_type = 'image'

    elif ext in vid_ext_list:
        source_type = 'video'

    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)

elif 'usb' in img_source:

    source_type = 'usb'
    usb_idx = int(img_source[3:])

elif 'picamera' in img_source:

    source_type = 'picamera'
    picam_idx = int(img_source[8:])

else:

    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)


# ============================================================
# Parse user-specified processing resolution
# ============================================================

resize = False

if user_res:

    resize = True

    try:
        resW, resH = map(int, user_res.lower().split('x'))

    except ValueError:

        print('ERROR: Invalid --resolution format.')
        print('Use format such as: 640x480 or 1280x720')
        sys.exit(0)


# ============================================================
# Parse user-specified DISPLAY WINDOW resolution
# ============================================================

try:

    displayW, displayH = map(int, display_res.lower().split('x'))

except ValueError:

    print('ERROR: Invalid --display format.')
    print('Use format such as: 1280x720 or 1600x900')
    sys.exit(0)


if displayW <= 0 or displayH <= 0:

    print('ERROR: Display width and height must be greater than 0.')
    sys.exit(0)


# ============================================================
# Check if recording is valid and set up recording
# ============================================================

if record:

    if source_type not in ['video', 'usb']:

        print('Recording only works for video and camera sources.')
        sys.exit(0)

    if not user_res:

        print('Please specify resolution to record video at.')
        sys.exit(0)

    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30

    recorder = cv2.VideoWriter(
        record_name,
        cv2.VideoWriter_fourcc(*'MJPG'),
        record_fps,
        (resW, resH)
    )


# ============================================================
# Load or initialize image source
# ============================================================

if source_type == 'image':

    imgs_list = [img_source]


elif source_type == 'folder':

    imgs_list = []

    filelist = glob.glob(img_source + '/*')

    for file in filelist:

        _, file_ext = os.path.splitext(file)

        if file_ext in img_ext_list:
            imgs_list.append(file)


elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video':
        cap_arg = img_source

    elif source_type == 'usb':
        cap_arg = usb_idx

    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified
    if user_res:

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)


elif source_type == 'picamera':

    from picamera2 import Picamera2

    cap = Picamera2()

    cap.configure(
        cap.create_video_configuration(
            main={
                "format": 'RGB888',
                "size": (resW, resH)
            }
        )
    )

    cap.start()


# ============================================================
# Create RESIZABLE output display window
# ============================================================

WINDOW_NAME = 'YOLO segmentation results'

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

# Set initial window size
cv2.resizeWindow(
    WINDOW_NAME,
    displayW,
    displayH
)


# ============================================================
# Set mask colors
# ============================================================

mask_colors = [
    (164, 120, 87),
    (68, 148, 228),
    (93, 97, 209),
    (178, 182, 133),
    (88, 159, 106),
    (96, 202, 231),
    (159, 124, 168),
    (169, 162, 241),
    (98, 118, 150),
    (172, 176, 184)
]


# ============================================================
# Transparency factor for mask fill
# ============================================================

mask_alpha = 0.4


# ============================================================
# Initialize control and status variables
# ============================================================

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0


# ============================================================
# Begin inference loop
# ============================================================

while True:

    t_start = time.perf_counter()


    # ========================================================
    # Load frame from image source
    # ========================================================

    if source_type == 'image' or source_type == 'folder':

        if img_count >= len(imgs_list):

            print('All images have been processed. Exiting program.')
            break

        img_filename = imgs_list[img_count]

        frame = cv2.imread(img_filename)

        img_count = img_count + 1


    elif source_type == 'video':

        ret, frame = cap.read()

        if not ret:

            print('Reached end of the video file. Exiting program.')
            break


    elif source_type == 'usb':

        ret, frame = cap.read()

        if (frame is None) or (not ret):

            print(
                'Unable to read frames from the camera. '
                'This indicates the camera is disconnected or not working. '
                'Exiting program.'
            )

            break


    elif source_type == 'picamera':

        frame = cap.capture_array()

        if frame is None:

            print(
                'Unable to read frames from the Picamera. '
                'This indicates the camera is disconnected or not working. '
                'Exiting program.'
            )

            break


    # ========================================================
    # Resize FRAME if --resolution was specified
    # ========================================================

    if resize:

        frame = cv2.resize(
            frame,
            (resW, resH)
        )


    # ========================================================
    # Run YOLOv11s segmentation inference
    # ========================================================

    results = model(
        frame,
        verbose=False,
        retina_masks=True
    )


    # ========================================================
    # Extract results
    # ========================================================

    detections = results[0].boxes
    masks = results[0].masks


    # ========================================================
    # Initialize object counter
    # ========================================================

    object_count = 0


    # ========================================================
    # Create overlay for transparent mask
    # ========================================================

    overlay = frame.copy()


    # ========================================================
    # Go through each detection
    # ========================================================

    if masks is not None:

        # masks.xy contains polygon coordinates
        mask_polygons = masks.xy

        for i in range(len(detections)):

            # ------------------------------------------------
            # Get class ID and class name
            # ------------------------------------------------

            classidx = int(
                detections[i].cls.item()
            )

            classname = labels[classidx]


            # ------------------------------------------------
            # Get confidence
            # ------------------------------------------------

            conf = detections[i].conf.item()


            # ------------------------------------------------
            # Only display detections above threshold
            # ------------------------------------------------

            if conf > min_thresh:

                color = mask_colors[
                    classidx % 10
                ]


                # ------------------------------------------------
                # Get polygon
                # ------------------------------------------------

                polygon = mask_polygons[i]

                if polygon is None or len(polygon) == 0:

                    continue

                polygon = polygon.astype(
                    np.int32
                )


                # ------------------------------------------------
                # Draw filled segmentation mask
                # ------------------------------------------------

                cv2.fillPoly(
                    overlay,
                    [polygon],
                    color
                )


                # ------------------------------------------------
                # Draw mask outline
                # ------------------------------------------------

                cv2.polylines(
                    frame,
                    [polygon],
                    isClosed=True,
                    color=color,
                    thickness=2
                )


                # ------------------------------------------------
                # Calculate object center
                # ------------------------------------------------

                x_coords = polygon[:, 0]
                y_coords = polygon[:, 1]

                cx = int(
                    (x_coords.min() + x_coords.max()) / 2
                )

                cy = int(
                    (y_coords.min() + y_coords.max()) / 2
                )


                # ------------------------------------------------
                # Create label
                # ------------------------------------------------

                label = f'{classname}: {int(conf * 100)}%'

                font_scale = 0.4
                font_thickness = 1

                labelSize, baseLine = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_thickness
                )


                # ------------------------------------------------
                # Position label
                # ------------------------------------------------

                text_x = cx - labelSize[0] // 2
                text_y = cy + labelSize[1] // 2


                # ------------------------------------------------
                # Draw label background
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (
                        text_x - 2,
                        text_y - labelSize[1] - 2
                    ),
                    (
                        text_x + labelSize[0] + 2,
                        text_y + baseLine
                    ),
                    color,
                    cv2.FILLED
                )


                # ------------------------------------------------
                # Draw label text
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    font_thickness
                )


                # ------------------------------------------------
                # Increase object count
                # ------------------------------------------------

                object_count += 1


    # ========================================================
    # Draw black information box
    # ========================================================

    cv2.rectangle(
        overlay,
        (0, 0),
        (270, 55),
        (0, 0, 0),
        cv2.FILLED
    )


    # ========================================================
    # Blend segmentation overlay
    # ========================================================

    frame = cv2.addWeighted(
        overlay,
        mask_alpha,
        frame,
        1 - mask_alpha,
        0
    )


    # ========================================================
    # Calculate and draw FPS
    # ========================================================

    if source_type in ['video', 'usb', 'picamera']:

        cv2.putText(
            frame,
            f'FPS: {avg_frame_rate:0.2f}',
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (54, 224, 54),
            2
        )


    # ========================================================
    # Draw object count
    # ========================================================

    cv2.putText(
        frame,
        f'Number of objects: {object_count}',
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (54, 224, 54),
        2
    )


    # ========================================================
    # Display segmentation results
    #
    # IMPORTANT:
    # The window is RESIZABLE.
    # You can manually drag its edges/corners.
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # Record results if requested
    # ========================================================

    if record:

        recorder.write(frame)


    # ========================================================
    # Keyboard controls
    # ========================================================

    if source_type == 'image' or source_type == 'folder':

        key = cv2.waitKey(0)

    else:

        key = cv2.waitKey(5)


    # --------------------------------------------------------
    # Q = Quit
    # --------------------------------------------------------

    if key == ord('q') or key == ord('Q'):

        break


    # --------------------------------------------------------
    # S = Pause
    # --------------------------------------------------------

    elif key == ord('s') or key == ord('S'):

        cv2.waitKey()


    # --------------------------------------------------------
    # P = Save current frame
    # --------------------------------------------------------

    elif key == ord('p') or key == ord('P'):

        cv2.imwrite(
            'capture.png',
            frame
        )


    # ========================================================
    # Calculate FPS for current frame
    # ========================================================

    t_stop = time.perf_counter()

    frame_rate_calc = float(
        1 / (t_stop - t_start)
    )


    # ========================================================
    # Store FPS
    # ========================================================

    if len(frame_rate_buffer) >= fps_avg_len:

        frame_rate_buffer.pop(0)

        frame_rate_buffer.append(
            frame_rate_calc
        )

    else:

        frame_rate_buffer.append(
            frame_rate_calc
        )


    # ========================================================
    # Calculate average FPS
    # ========================================================

    avg_frame_rate = np.mean(
        frame_rate_buffer
    )


# ============================================================
# Clean up
# ============================================================

print(
    f'Average pipeline FPS: {avg_frame_rate:.2f}'
)


if source_type == 'video' or source_type == 'usb':

    cap.release()


elif source_type == 'picamera':

    cap.stop()


if record:

    recorder.release()


cv2.destroyAllWindows()