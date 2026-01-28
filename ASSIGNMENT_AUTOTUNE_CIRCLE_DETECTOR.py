import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class ContourCircleDetector:
    def __init__(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image not found")
        if os.path.splitext(image_path) in ('.png','.jpeg','.jpg','.bmp'):
            raise ValueError("File not of appropriate type")
        self.image = cv2.imread(image_path)
        self.gray = None
        self.edges = None
        self.contours = None

    # -----------------------------
    # STEP 1: PREPROCESS
    # -----------------------------
    def preprocess(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        gray = cv2.equalizeHist(gray)
        self.gray = gray
        return gray

    # -----------------------------
    # STEP 2: EDGE DETECTION
    # -----------------------------
    def detect_edges(self):
        edges = cv2.Canny(self.gray,50, 120)
        self.edges = edges
        return edges

    # -----------------------------
    # STEP 3: FIND CONTOURS
    # -----------------------------
    def find_contours(self):
        contours, _ = cv2.findContours(
            self.edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        self.contours = contours
        return contours

    # -----------------------------
    # STEP 4: SCORE CONTOURS
    # -----------------------------
    def score_contour(self, cnt):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area < 50 or perimeter == 0:
            return None

        # 1️⃣ Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # 2️⃣ Fit circle
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * r * r
        area_ratio = area / circle_area

        # 3️⃣ Edge support on ring
        ring_mask = np.zeros(self.edges.shape, dtype=np.uint8)
        cv2.circle(ring_mask, (int(x), int(y)), int(r), 255, 2)

        edge_support = np.count_nonzero(
            cv2.bitwise_and(self.edges, ring_mask)
        )
        support_ratio = edge_support / (2 * np.pi * r)

        # ❌ Reject early
        if circularity < 0.5 or support_ratio < 0.2:
            return None

        # 🔢 Final score
        score = (
            0.5 * circularity +
            0.3 * support_ratio +
            0.2 * area_ratio
        )

        return {
            "center": (int(x), int(y)),
            "radius": int(r),
            "score": score
        }

    # -----------------------------
    # STEP 5: DETECT CIRCLES
    # -----------------------------
    def detect_circles(self, score_threshold=0.6):
        detected = []

        for cnt in self.contours:
            result = self.score_contour(cnt)
            if result and result["score"] >= score_threshold:
                detected.append(result)

        return detected

    # -----------------------------
    # STEP 6: DRAW RESULTS
    # -----------------------------
    def draw(self, circles):
        output = self.image.copy()

        for c in circles:
            x, y = c["center"]
            r = c["radius"]
            score = c["score"]

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.putText(
                output,
                f"{score:.2f}",
                (x - 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        return output

    # -----------------------------
    # FULL PIPELINE
    # -----------------------------
    def run(self):
        self.preprocess()
        self.detect_edges()
        self.find_contours()
        circles = self.detect_circles()
        output = self.draw(circles)
        return output, circles


# -----------------------------
# USAGE
# -----------------------------
if __name__ == "__main__":
    image_path = input("Enter image path:")   # CHANGE THIS

    detector = ContourCircleDetector(image_path)
    result, circles = detector.run()

    print("Detected circles:", circles)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()



            
        
            
            
            
            
        
        
            
            
            
            
        