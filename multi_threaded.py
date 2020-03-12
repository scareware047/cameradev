"""
Author: Tejas Pandey
"""

import cv2
import time
import threading
import argparse


class AsyncVideoCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.running = False
        self.read_lock = threading.Lock()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

    def start(self):
        if self.running:
            return None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=0)
    args = parser.parse_args()

    cap = AsyncVideoCapture(args.src)
    cap.start()
    fps = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()

    while True:
        _, frame = cap.read()

        curr_time = time.time() - start
        frame = cv2.resize(frame, (720, 480))
        cv2.putText(frame, 'FPS: {}'.format(round(fps/curr_time)),
                    (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Multi-Threaded", frame)
        fps += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Total Frames: {}".format(fps))
    print("Time Taken: {:.2f}".format(time.time()-start))
    print("FPS: {}".format(round(fps/curr_time)))
    cap.release()
    cv2.destroyAllWindows()
