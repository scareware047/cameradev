"""
Author: Tejas Pandey
"""

import cv2
import time

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    fps = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()

    while True:
        _, frame = cap.read()

        curr_time = time.time() - start
        cv2.putText(frame, 'FPS: {}'.format(round(fps/curr_time)),
                    (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Display", frame)
        fps += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
