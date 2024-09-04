import cv2
import numpy as np
import os
import subprocess
from datetime import datetime

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press SPACE to capture', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def select_roi(image):
    roi = cv2.selectROI("Select area to crop", image)
    cv2.destroyAllWindows()
    return roi

def crop_image(image, roi):
    return image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

def save_image(image, filename):
    cv2.imwrite(filename, image)

def convert_to_pbm(input_file, output_file):
    subprocess.run(['convert', input_file, '-threshold', '30%', output_file])

def convert_to_svg(input_file, output_file):
    subprocess.run(['potrace', '-s', '-o', output_file, input_file])

def main():
    # Create results folder if it doesn't exist
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Capture image
    image = capture_image()

    # Select ROI and crop
    roi = select_roi(image)
    cropped_image = crop_image(image, roi)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = input("What diagram is this? \n")

    # Save cropped image
    jpg_filename = os.path.join(results_folder, f"{base_filename}.jpg")
    save_image(cropped_image, jpg_filename)

    # Convert to PBM
    pbm_filename = os.path.join(results_folder, f"{base_filename}.pbm")
    convert_to_pbm(jpg_filename, pbm_filename)

    # Convert to SVG
    svg_filename = os.path.join(results_folder, f"{base_filename}.svg")
    convert_to_svg(pbm_filename, svg_filename)

    print(f"SVG file saved as: {svg_filename}")

if __name__ == "__main__":
    main()