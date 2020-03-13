"""
Author: Tejas Pandey
"""

import cv2
import time
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.src)
    fps = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()

    while True:
        _, frame = cap.read()

        curr_time = time.time() - start
        frame = cv2.resize(frame, (720, 480))
        cv2.putText(frame, 'FPS: {}'.format(round(fps/curr_time)),
                    (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Default", frame)
        fps += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Total Frames: {}".format(fps))
    print("Time Taken: {:.2f}".format(time.time()-start))
    print("FPS: {}".format(round(fps/curr_time)))
    cap.release()
    cv2.destroyAllWindows()
