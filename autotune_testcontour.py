import cv2
import numpy as np
image_path=input("Enter image path")
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(
    blur, cv2.HOUGH_GRADIENT,
    dp=1.2, minDist=50,
    param1=100, param2=40,
    minRadius=10, maxRadius=0
)

refined = img.copy()

if circles is not None:
    circles = np.uint16(np.around(circles[0])).astype(int)
    for (cx, cy, r) in circles:
        # ROI around circle
        pad = int(r * 1.2)
        x1, y1 = max(cx - pad, 0), max(cy - pad, 0)
        x2, y2 = min(cx + pad, img.shape[1]-1), min(cy + pad, img.shape[0]-1)

        roi = gray[y1:y2, x1:x2]

        # Edge / binary mask
        edges = cv2.Canny(roi, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Pick contour whose centroid is closest to Hough center
        best_cnt = None
        best_dist = 1e9
        for cnt in contours:
            (mx, my), _ = cv2.minEnclosingCircle(cnt)
            # map back to full-image coords
            gx, gy = x1 + mx, y1 + my
            d = (gx - cx)**2 + (gy - cy)**2
            if d < best_dist:
                best_dist = d
                best_cnt = cnt

        if best_cnt is None:
            continue

        # Fit circle to best contour
        (fx, fy), fr = cv2.minEnclosingCircle(best_cnt)
        fx, fy = int(x1 + fx), int(y1 + fy)
        fr = int(fr)

        cv2.circle(refined, (fx, fy), fr, (0, 0, 255), 2)  # refined circle

cv2.imshow("Refined circles", refined)
cv2.waitKey(0)
