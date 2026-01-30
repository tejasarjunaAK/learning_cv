# Image and Video Processing Toolkit

This repository contains three Python projects for image and video processing:

- Circle Detection with Automatic Parameter Tuning  
- Image Augmentation  
- Video Circle Detection  

Each project is independent but demonstrates advanced image processing techniques using OpenCV, NumPy, and Matplotlib.

---

## 1. Circle Detection (autotune_my_method)

### Description:
Automatically detects circles in an image and refines them using contour analysis. It selects the best detection parameter using Canny edge scoring and classifies detected circles by size.

### Features:

- Automatic circle detection using Hough Transform.
- Contour-based refinement for accurate circle detection.
- Size classification: small, middle, or large.
- Visualization of original vs processed images.

### Output:

- Displays original and processed image.
- Annotates circles with size labels (small/middle/large).

---

## 2. Image Augmentation (augmentation)

### Description:
Performs a variety of image augmentations for dataset enhancement and experimentation.

### Features:

- Brightness and contrast adjustment.
- Rotation (-40° to +40°).
- Saturation modification.
- Gaussian noise addition.
- Translation, zooming, and flipping (horizontal, vertical, both).
- Saves augmented images with descriptive filenames.
- Displays augmentations in a 2x4 Matplotlib grid.

### Output:

Saves augmented images in the current directory:

- `brightness_augmented_image.jpg`  
- `contrast_augmented_image.jpg`  
- `rotated_augmented_image.jpg`  
- `saturation_augmented_image.jpg`  
- `gaussian_noise_augmented_image.jpg`  
- `translated_image.png`  
- `zoomed_image.jpg`  
- `horizontal_flipped_image.jpg`  
- `vertical_flipped_image.jpg`  
- `both_flipped_image.jpg`  

Shows a preview of augmentations.

---

## 3. Video Circle Detection (video_circle)

### Description:
Detects circles in video frames using Hough Transform and applies scoring to select optimal detection parameters. The processed video is saved with detected circles.

### Features:

- Reads input video and processes each frame.
- Detects circles with optimized parameters.
- Writes output video with detected circles overlaid.
- Automatically handles frame-by-frame detection.

### Output:

- Saves a processed video (`circle_detector.mp4`) with circles detected on each frame.  
- Prints the best parameter value for circle detection.

---

## Repository Structure

.
├── ASSIGNMENT_CIRCLE_DETECTOR.py # Circle detection script
├── ASSIGNMENT_AUGMENTATION.py # Image augmentation script
├── ASSIGNMENT_CIRCLE_DETECTION_VIDEO.py # Video circle detection script
├── ASSIGNMENT_AUTOTUNING_PART2.py #Autotuned circle detection script
├── README.md # This file

---

## Challenges and Solutions

For autotuning the HoughCircles parameters and for the image to only detect actual circlea and avoid false circles, series of strategies were used. First, all parameters except param2(param1,minDist,minRadius,maxRadius) were kept fixed and to values that will not support false circles formation. But the main parameter -param2 was changed from 50 -80 through intervals of 2. Then for each param2 value penalties were assigned based on two criteria. penaltyA: if more than 12 circles detected then penalty proportional to number of circles was assigned and penaltyB. For second penalty, we create canny of image with detected circles using each param2 values and then check how much of intensity differnece is caused by the detected circle edges over the dilated canny image of original image. Then the two penalties are added. The param2 value which had the least penalties were given high score. Hence best param2 is chosen. Even with best param2 there might be false circles as HoughCircles draws circles usng broken arcs , so around actual circle also multiple false circles appear. Therefore we go through each detected circle, consider only the region just around the circles, avoid minor contours and choose the contour closest to circle that is large and draw circle using it. This reduces false circles by great number.

---

## Scope For Improvement:

Choosing contours close to the center in the last step may give noisy large contours also so small circles inside actual circles maybe detected without the main circle being detected.

---

## Author

Tejas Arjuna Ashok Kumar



