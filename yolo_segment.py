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

parser = argparse.ArgumentParser(
    description='YOLOv11s Segmentation + Defect Measurement'
)


parser.add_argument(
    '--model',
    help='Path to YOLOv11s segmentation model file '
         '(example: "runs/segment/train/weights/best.pt")',
    required=True
)


parser.add_argument(
    '--source',
    help='Image source: image file, image folder, video file, '
         'USB camera (usb0), or Picamera (picamera0)',
    required=True
)


parser.add_argument(
    '--thresh',
    help='Minimum confidence threshold '
         '(example: 0.4)',
    default=0.5
)


parser.add_argument(
    '--resolution',
    help='Processing resolution in WxH '
         '(example: 640x480). '
         'If omitted, original source resolution is used.',
    default=None
)


parser.add_argument(
    '--display',
    help='Initial output window size in WxH '
         '(example: 1280x720)',
    default='1280x720'
)


parser.add_argument(
    '--scale',
    type=float,
    default=None,
    help='Physical scale in mm per pixel. '
         'Example: --scale 0.1 means 1 pixel = 0.1 mm.'
)


parser.add_argument(
    '--record',
    help='Record results from video/webcam. '
         'Requires --resolution.',
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
physical_scale = args.scale
record = args.record


# ============================================================
# Validate physical scale
# ============================================================

if physical_scale is not None:

    if physical_scale <= 0:

        print('ERROR: --scale must be greater than 0.')

        sys.exit(0)

    print()
    print('============================================================')
    print('Physical Measurement Enabled')
    print('============================================================')
    print(f'1 pixel = {physical_scale} mm')
    print(f'1 pixel² = {physical_scale ** 2} mm²')
    print('============================================================')
    print()


else:

    print()
    print('============================================================')
    print('Physical Measurement NOT Calibrated')
    print('============================================================')
    print('Measurements will be displayed in pixels.')
    print()
    print('To enable mm measurements, use for example:')
    print('    --scale 0.1')
    print('============================================================')
    print()


# ============================================================
# Check model path
# ============================================================

if not os.path.exists(model_path):

    print(
        'ERROR: Model path is invalid or model was not found.'
    )

    print(
        'Make sure the model filename was entered correctly.'
    )

    sys.exit(0)


# ============================================================
# Load YOLOv11s segmentation model
# ============================================================

print()
print('Loading YOLOv11s segmentation model...')

model = YOLO(
    model_path,
    task='segment'
)

labels = model.names

print('YOLOv11s model loaded successfully.')


# ============================================================
# Determine source type
# ============================================================

img_ext_list = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.bmp',
    '.BMP'
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

        print(
            f'File extension {ext} is not supported.'
        )

        sys.exit(0)


elif 'usb' in img_source:

    source_type = 'usb'

    usb_idx = int(
        img_source[3:]
    )


elif 'picamera' in img_source:

    source_type = 'picamera'

    picam_idx = int(
        img_source[8:]
    )


else:

    print(
        f'Input {img_source} is invalid.'
    )

    sys.exit(0)


# ============================================================
# Parse processing resolution
# ============================================================

resize = False


if user_res:

    resize = True

    try:

        resW, resH = map(
            int,
            user_res.lower().split('x')
        )

    except ValueError:

        print(
            'ERROR: Invalid --resolution format.'
        )

        print(
            'Use format such as: 640x480'
        )

        sys.exit(0)


# ============================================================
# Parse display window resolution
# ============================================================

try:

    displayW, displayH = map(
        int,
        display_res.lower().split('x')
    )

except ValueError:

    print(
        'ERROR: Invalid --display format.'
    )

    print(
        'Use format such as: 1280x720'
    )

    sys.exit(0)


if displayW <= 0 or displayH <= 0:

    print(
        'ERROR: Display dimensions must be greater than 0.'
    )

    sys.exit(0)


# ============================================================
# Recording setup
# ============================================================

if record:

    if source_type not in ['video', 'usb']:

        print(
            'Recording only works for video and camera sources.'
        )

        sys.exit(0)


    if not user_res:

        print(
            'Please specify --resolution when recording.'
        )

        sys.exit(0)


    record_name = 'measurement_result.avi'

    record_fps = 30


    recorder = cv2.VideoWriter(
        record_name,
        cv2.VideoWriter_fourcc(*'MJPG'),
        record_fps,
        (resW, resH)
    )


# ============================================================
# Load image source
# ============================================================

if source_type == 'image':

    imgs_list = [
        img_source
    ]


elif source_type == 'folder':

    imgs_list = []

    filelist = glob.glob(
        img_source + '/*'
    )

    for file in filelist:

        _, file_ext = os.path.splitext(
            file
        )

        if file_ext in img_ext_list:

            imgs_list.append(
                file
            )

    # Sort images for consistent order
    imgs_list.sort()


elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video':

        cap_arg = img_source

    else:

        cap_arg = usb_idx


    cap = cv2.VideoCapture(
        cap_arg
    )


    if user_res:

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            resW
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            resH
        )


elif source_type == 'picamera':

    from picamera2 import Picamera2


    cap = Picamera2()


    cap.configure(
        cap.create_video_configuration(
            main={
                'format': 'RGB888',
                'size': (
                    resW,
                    resH
                )
            }
        )
    )


    cap.start()


# ============================================================
# Create resizable display window
# ============================================================

WINDOW_NAME = 'YOLOv11s Segmentation + Measurement'


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    WINDOW_NAME,
    displayW,
    displayH
)


# ============================================================
# Mask colors
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
# Mask transparency
# ============================================================

mask_alpha = 0.4


# ============================================================
# Status variables
# ============================================================

avg_frame_rate = 0

frame_rate_buffer = []

fps_avg_len = 200

img_count = 0


# ============================================================
# Measurement helper function
# ============================================================

def calculate_measurements(polygon, scale=None):
    """
    Calculate geometric measurements from a segmentation polygon.

    Returns:
        area_pixels
        perimeter_pixels
        bbox_width_pixels
        bbox_height_pixels
        length_pixels
        width_pixels
        area_mm2
        perimeter_mm
        length_mm
        width_mm
    """

    # --------------------------------------------------------
    # Convert polygon to OpenCV contour format
    # --------------------------------------------------------

    contour = polygon.reshape(
        (-1, 1, 2)
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area_pixels = cv2.contourArea(
        contour
    )


    # --------------------------------------------------------
    # Perimeter
    # --------------------------------------------------------

    perimeter_pixels = cv2.arcLength(
        contour,
        True
    )


    # --------------------------------------------------------
    # Axis-aligned bounding rectangle
    # --------------------------------------------------------

    x, y, bbox_width, bbox_height = cv2.boundingRect(
        contour.astype(np.int32)
    )


    # --------------------------------------------------------
    # Minimum-area rotated rectangle
    #
    # This gives a better estimate for elongated defects
    # such as scratches.
    # --------------------------------------------------------

    rect = cv2.minAreaRect(
        contour
    )


    rect_width, rect_height = rect[1]


    # Long dimension = approximate defect length
    # Short dimension = approximate defect width

    length_pixels = max(
        rect_width,
        rect_height
    )


    width_pixels = min(
        rect_width,
        rect_height
    )


    # --------------------------------------------------------
    # Physical measurements
    # --------------------------------------------------------

    if scale is not None:

        area_mm2 = (
            area_pixels *
            (scale ** 2)
        )


        perimeter_mm = (
            perimeter_pixels *
            scale
        )


        length_mm = (
            length_pixels *
            scale
        )


        width_mm = (
            width_pixels *
            scale
        )


    else:

        area_mm2 = None

        perimeter_mm = None

        length_mm = None

        width_mm = None


    return {

        'area_pixels': area_pixels,

        'perimeter_pixels': perimeter_pixels,

        'bbox_width_pixels': bbox_width,

        'bbox_height_pixels': bbox_height,

        'length_pixels': length_pixels,

        'width_pixels': width_pixels,

        'area_mm2': area_mm2,

        'perimeter_mm': perimeter_mm,

        'length_mm': length_mm,

        'width_mm': width_mm,

        'bbox_x': x,

        'bbox_y': y

    }


# ============================================================
# Format measurement text
# ============================================================

def create_measurement_lines(
    measurements
):

    if physical_scale is not None:

        lines = [

            f"Area: {measurements['area_mm2']:.2f} mm2",

            f"Length: {measurements['length_mm']:.2f} mm",

            f"Width: {measurements['width_mm']:.2f} mm",

            f"Perimeter: {measurements['perimeter_mm']:.2f} mm"

        ]

    else:

        lines = [

            f"Area: {measurements['area_pixels']:.0f} px2",

            f"Length: {measurements['length_pixels']:.1f} px",

            f"Width: {measurements['width_pixels']:.1f} px",

            f"Perimeter: {measurements['perimeter_pixels']:.1f} px"

        ]


    return lines


# ============================================================
# Begin inference loop
# ============================================================

while True:

    t_start = time.perf_counter()


    # ========================================================
    # Load frame
    # ========================================================

    if source_type == 'image' or source_type == 'folder':

        if img_count >= len(imgs_list):

            print(
                'All images have been processed.'
            )

            break


        img_filename = imgs_list[
            img_count
        ]


        frame = cv2.imread(
            img_filename
        )


        if frame is None:

            print(
                f'WARNING: Unable to read {img_filename}'
            )

            img_count += 1

            continue


        img_count += 1


    elif source_type == 'video':

        ret, frame = cap.read()


        if not ret:

            print(
                'Reached end of video.'
            )

            break


    elif source_type == 'usb':

        ret, frame = cap.read()


        if (
            frame is None
            or not ret
        ):

            print(
                'Unable to read frame from camera.'
            )

            break


    elif source_type == 'picamera':

        frame = cap.capture_array()


        if frame is None:

            print(
                'Unable to read frame from Picamera.'
            )

            break


    # ========================================================
    # Resize frame for inference if requested
    # ========================================================

    if resize:

        frame = cv2.resize(
            frame,
            (
                resW,
                resH
            )
        )


    # ========================================================
    # YOLOv11s segmentation inference
    # ========================================================

    results = model(
        frame,
        verbose=False,
        retina_masks=True
    )


    # ========================================================
    # Extract detections and masks
    # ========================================================

    detections = results[0].boxes

    masks = results[0].masks


    # ========================================================
    # Object counter
    # ========================================================

    object_count = 0


    # ========================================================
    # Store measurement information
    # ========================================================

    measurement_results = []


    # ========================================================
    # Create mask overlay
    # ========================================================

    overlay = frame.copy()


    # ========================================================
    # Process segmentation masks
    # ========================================================

    if masks is not None:

        mask_polygons = masks.xy


        for i in range(
            len(detections)
        ):


            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            classidx = int(
                detections[i].cls.item()
            )


            classname = labels[
                classidx
            ]


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            conf = detections[i].conf.item()


            # ------------------------------------------------
            # Confidence filtering
            # ------------------------------------------------

            if conf <= min_thresh:

                continue


            # ------------------------------------------------
            # Color
            # ------------------------------------------------

            color = mask_colors[
                classidx % len(mask_colors)
            ]


            # ------------------------------------------------
            # Get segmentation polygon
            # ------------------------------------------------

            polygon = mask_polygons[i]


            if (
                polygon is None
                or len(polygon) < 3
            ):

                continue


            polygon = polygon.astype(
                np.int32
            )


            # =================================================
            # MEASUREMENT
            # =================================================

            measurements = calculate_measurements(
                polygon,
                physical_scale
            )


            # Add identification information

            measurements['id'] = (
                object_count + 1
            )

            measurements['class'] = (
                classname
            )

            measurements['confidence'] = (
                conf
            )


            measurement_results.append(
                measurements
            )


            object_count += 1


            # =================================================
            # Draw segmentation mask
            # =================================================

            cv2.fillPoly(
                overlay,
                [polygon],
                color
            )


            # =================================================
            # Draw segmentation outline
            # =================================================

            cv2.polylines(
                frame,
                [polygon],
                True,
                color,
                2
            )


            # =================================================
            # Bounding rectangle
            # =================================================

            x = measurements[
                'bbox_x'
            ]

            y = measurements[
                'bbox_y'
            ]

            bbox_width = measurements[
                'bbox_width_pixels'
            ]

            bbox_height = measurements[
                'bbox_height_pixels'
            ]


            cv2.rectangle(
                frame,
                (
                    x,
                    y
                ),
                (
                    x + bbox_width,
                    y + bbox_height
                ),
                color,
                1
            )


            # =================================================
            # Calculate center
            # =================================================

            x_coords = polygon[
                :, 0
            ]

            y_coords = polygon[
                :, 1
            ]


            cx = int(
                (
                    x_coords.min()
                    +
                    x_coords.max()
                ) / 2
            )


            cy = int(
                (
                    y_coords.min()
                    +
                    y_coords.max()
                ) / 2
            )


            # =================================================
            # Create measurement text
            # =================================================

            measurement_lines = (
                create_measurement_lines(
                    measurements
                )
            )


            # =================================================
            # Build label
            # =================================================

            label = (
                f"#{object_count} "
                f"{classname} "
                f"{conf * 100:.1f}%"
            )


            # =================================================
            # Determine text box size
            # =================================================

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.45

            font_thickness = 1

            line_height = 18


            all_text = [
                label
            ] + measurement_lines


            max_text_width = 0


            for text in all_text:

                text_size, _ = cv2.getTextSize(
                    text,
                    font,
                    font_scale,
                    font_thickness
                )


                max_text_width = max(
                    max_text_width,
                    text_size[0]
                )


            box_width = (
                max_text_width + 10
            )


            box_height = (
                len(all_text)
                *
                line_height
                +
                8
            )


            # =================================================
            # Keep text box inside image
            # =================================================

            frame_height, frame_width = (
                frame.shape[:2]
            )


            text_box_x = max(
                0,
                min(
                    cx - box_width // 2,
                    frame_width - box_width - 1
                )
            )


            # Prefer placing above defect

            text_box_y = (
                cy
                - box_height
                - 10
            )


            # If there isn't enough room above,
            # place below the defect.

            if text_box_y < 0:

                text_box_y = min(
                    cy + 10,
                    frame_height - box_height - 1
                )


            # =================================================
            # Draw measurement information background
            # =================================================

            cv2.rectangle(
                frame,
                (
                    text_box_x,
                    text_box_y
                ),
                (
                    text_box_x + box_width,
                    text_box_y + box_height
                ),
                color,
                cv2.FILLED
            )


            # =================================================
            # Draw measurement text
            # =================================================

            for line_index, text in enumerate(
                all_text
            ):

                text_x = (
                    text_box_x + 5
                )

                text_y = (
                    text_box_y
                    + 16
                    + line_index * line_height
                )


                cv2.putText(
                    frame,
                    text,
                    (
                        text_x,
                        text_y
                    ),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness,
                    cv2.LINE_AA
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
    # Draw FPS
    # ========================================================

    if source_type in [
        'video',
        'usb',
        'picamera'
    ]:

        cv2.putText(
            frame,
            f'FPS: {avg_frame_rate:.2f}',
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (54, 224, 54),
            2
        )


    # ========================================================
    # Draw total defect count
    # ========================================================

    cv2.putText(
        frame,
        f'Segmented Defects: {object_count}',
        (10, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (54, 224, 54),
        2
    )


    # ========================================================
    # Display results
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # Record
    # ========================================================

    if record:

        recorder.write(
            frame
        )


    # ========================================================
    # Keyboard controls
    # ========================================================

    if source_type in [
        'image',
        'folder'
    ]:

        key = cv2.waitKey(0)

    else:

        key = cv2.waitKey(5)


    # --------------------------------------------------------
    # Q = Quit
    # --------------------------------------------------------

    if key in [
        ord('q'),
        ord('Q')
    ]:

        break


    # --------------------------------------------------------
    # S = Pause
    # --------------------------------------------------------

    elif key in [
        ord('s'),
        ord('S')
    ]:

        cv2.waitKey(0)


    # --------------------------------------------------------
    # P = Save result
    # --------------------------------------------------------

    elif key in [
        ord('p'),
        ord('P')
    ]:

        cv2.imwrite(
            'measurement_capture.png',
            frame
        )

        print(
            'Saved: measurement_capture.png'
        )


    # ========================================================
    # Print measurement information to console
    # ========================================================

    if source_type in [
        'image',
        'folder'
    ]:

        print()
        print(
            '============================================================'
        )

        print(
            f'Image: {img_count}'
        )

        print(
            f'Detected/Segmented Defects: {object_count}'
        )

        print(
            '============================================================'
        )


        for result in measurement_results:

            print(
                f"Defect #{result['id']}"
            )

            print(
                f"  Class       : {result['class']}"
            )

            print(
                f"  Confidence  : "
                f"{result['confidence'] * 100:.2f}%"
            )


            print(
                f"  Area        : "
                f"{result['area_pixels']:.2f} px²"
            )

            print(
                f"  Perimeter   : "
                f"{result['perimeter_pixels']:.2f} px"
            )

            print(
                f"  Length      : "
                f"{result['length_pixels']:.2f} px"
            )

            print(
                f"  Width       : "
                f"{result['width_pixels']:.2f} px"
            )


            if physical_scale is not None:

                print(
                    f"  Area        : "
                    f"{result['area_mm2']:.2f} mm²"
                )

                print(
                    f"  Perimeter   : "
                    f"{result['perimeter_mm']:.2f} mm"
                )

                print(
                    f"  Length      : "
                    f"{result['length_mm']:.2f} mm"
                )

                print(
                    f"  Width       : "
                    f"{result['width_mm']:.2f} mm"
                )


            print(
                '------------------------------------------------------------'
            )


    # ========================================================
    # Calculate FPS
    # ========================================================

    t_stop = time.perf_counter()


    elapsed_time = (
        t_stop - t_start
    )


    if elapsed_time > 0:

        frame_rate_calc = (
            1.0 / elapsed_time
        )

    else:

        frame_rate_calc = 0


    # ========================================================
    # Store FPS
    # ========================================================

    if len(frame_rate_buffer) >= fps_avg_len:

        frame_rate_buffer.pop(0)


    frame_rate_buffer.append(
        frame_rate_calc
    )


    # ========================================================
    # Average FPS
    # ========================================================

    if len(frame_rate_buffer) > 0:

        avg_frame_rate = np.mean(
            frame_rate_buffer
        )


# ============================================================
# Cleanup
# ============================================================

print()
print(
    '============================================================'
)

print(
    f'Average pipeline FPS: '
    f'{avg_frame_rate:.2f}'
)

print(
    'Program finished.'
)

print(
    '============================================================'
)


if source_type in [
    'video',
    'usb'
]:

    cap.release()


elif source_type == 'picamera':

    cap.stop()


if record:

    recorder.release()


cv2.destroyAllWindows()