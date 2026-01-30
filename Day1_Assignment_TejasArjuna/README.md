This repository contains two image-processing projects taht convert a normal photograph into:

1)Pencil Sketch (Grayscale)

2)Color Pencil Sketch

Both projects use classic computer vision techniques such as image inversion, Gaussian blurring, image division, and Canny edge detection.

Projects Included
1️) Pencil Sketch (Grayscale)

Converts an image into a realistic pencil sketch

Enhances outlines using Canny edge detection

Displays original and sketch images side-by-side

Saves output images automatically

2️) Color Pencil Sketch

Produces a colored pencil sketch

Preserves color information using HSV color space

Controls brightness and saturation for artistic effect

Displays original and sketch using Matplotlib

Repository structure: 

Day1_Assignment_TejasArjuna/
├── pencil_sketch.py
├── ASSIGNMENT_COLOR_PENCIL_SKETCH.py
├── test_images
├── output_sketches
└── README.md

Algorithms Overview
1) Pencil Sketch Algorithm

Convert image to grayscale

Invert grayscale image

Apply Gaussian Blur

Invert blurred image

Divide grayscale by inverted blur

Apply Canny edge detection

Blend edges with pencil sketch

2) Color Sketch Algorithm

Generate pencil sketch (value channel)

Convert original image to HSV

Reduce saturation slightly

Replace V (brightness) channel with sketch

Convert back to RGB

Requirements

Install the required libraries:

pip install opencv-python numpy matplotlib

Error Handling

Invalid image formats are rejected

Missing files raise appropriate errors

Supported formats:

.png .jpg .jpeg .bmp

Concepts Used

Image inversion

Gaussian blur

Image division blending

Canny edge detection

HSV color manipulation

Weighted image blending

Challenges Faced && Solutions:

Traditional method of pencil sketch was not solving the problem of mild visibility of edges and boundaries when either side of the edge didn't have much of differnece in intensity. So a new method was thoght, -> The traditional method was followed and the intermediate output was then used to get edges using canny. This is done to make edges more dark when we have the edges and output image, we can blend them by giving each one weights. Now before that edges must be changed to black pixels before doing it. But the canny can't be inverted globally as the non edge background will( region in the intermediate output where intensity is high) will become where the intensity is even more higher and will dominate the image. Makng the whole image lighter or more whiter. So we conditionally invert by invertinng the pixels of very high intensity( >180) alone. So the intention is satisfied and then we blend these altered edges with the output obtained from traditionally created image. This gives image where we get darker edges than we could get from normal method.

Author

Tejas Arjuna Ashok Kumar
