import os
import cv2
import numpy as np

import db as card_db

class CardScanner:
    def __init__(self, database_dir="assets/database"):
        self.database_dir = database_dir
        # ORB must exist before the reference images are loaded (they are feature-extracted on load)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reference_database = self._load_database()

    def _load_database(self):
        reference = {}
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir, exist_ok=True)
            return reference
            
        for filename in os.listdir(self.database_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(self.database_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    card_id = os.path.splitext(filename)[0]
                    # Extract features
                    kp, des = self.orb.detectAndCompute(img, None)
                    # Card details come from the database (keyed by the image filename);
                    # images with no database row fall back to a name guessed from the filename.
                    info = card_db.get_card(card_id) or {
                        "name": card_id.replace("_", " ").title(),
                        "rarity": "Unknown",
                        "set": "Not in database",
                    }
                    reference[card_id] = {
                        "card_id": card_id,
                        "name": info["name"],
                        "rarity": info["rarity"],
                        "set": info["set"],
                        "image": img,
                        "kp": kp,
                        "des": des
                    }
        return reference

    def extract_card_contour(self, frame):
        """Finds the largest rectangular contour representing the card and warps it."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            area = cv2.contourArea(c)
            if area > 5000:  # Minimum area threshold
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    warped = self._four_point_transform(frame, approx.reshape(4, 2))
                    return warped, approx

        return None, None

    def _four_point_transform(self, image, pts):
        """Perspective transforms the quadrilateral card region into a flat rectangle."""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def match_card(self, warped_frame):
        """Matches warped frame against local DB using ORB feature matching."""
        if not self.reference_database:
            return {"card_id": "unknown", "name": "No Reference Cards in DB", "rarity": "N/A", "set": "N/A", "score": 0.0}

        gray_warp = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        kp_w, des_w = self.orb.detectAndCompute(gray_warp, None)

        if des_w is None:
            return {"card_id": "unknown", "name": "Unrecognized Card", "rarity": "N/A", "set": "N/A", "score": 0.0}

        best_match = None
        highest_score = 0

        for card_id, data in self.reference_database.items():
            if data["des"] is None:
                continue
            matches = self.bf.match(data["des"], des_w)
            score = len(matches)
            if score > highest_score:
                highest_score = score
                best_match = data

        if highest_score > 15: # Confidence threshold
            return {
                "card_id": best_match["card_id"],
                "name": best_match["name"],
                "rarity": best_match["rarity"],
                "set": best_match["set"],
                "score": float(highest_score)
            }
        
        return {"card_id": "unknown", "name": "Unknown Card", "rarity": "Unknown", "set": "Unknown", "score": float(highest_score)}
