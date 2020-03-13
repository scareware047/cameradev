"""
Script to run and display video stream using OpenCV.
Asynchronus implementation using queues.

Author: Tejas Pandey
"""

import cv2
import time
import threading
import argparse
from queue import Queue


class AsyncVideoCapture:
    """
    Asynchronus video capture.
    """

    def __init__(self, src):
        """
        Initialize camera capture from src.
        Creates a queue to keep all frames.

        Args:
            src: Input source.
        """

        # Load video capture from src.
        self.cap = cv2.VideoCapture(src)
        # Set current state to not running.
        self.running = False
        # Queue to store frames. Max size 256 frames.
        self.Q = Queue(256)

    def __exit__(self, exec_type, exc_value, traceback):
        """
        Release capture object on exit.
        """

        # Release video capture object.
        self.cap.release()

    def start(self):
        """
        Start streaming from camera.
        """

        # If already running don't start.
        if self.running:
            print("Already running!")
            return None
        # Set running to true.
        self.running = True
        # Create thread to push frames to queue.
        self.thread = threading.Thread(target=self.update, args=())
        # Set thread to daemon thread.
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        """
        Push frames to queue every read.
        """

        while self.running:
            # If Q not full read frame.
            if not self.Q.full():
                grabbed, frame = self.cap.read()
                self.Q.put([grabbed, frame])
            # Else put thread to sleep.
            else:
                time.sleep(0.1)

    def read(self):
        """
        Pop frames from queue.
        """

        return self.Q.get()

    def release(self):
        """
        Clean up and release memory.
        """

        self.running = False
        self.thread.join()
        self.cap.release()


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=0)
    parser.add_argument("--benchmark", action='store_true')
    args = parser.parse_args()

    # Initialize camera and metrics.
    cap = AsyncVideoCapture(args.src)
    cap.start()
    frames = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()

    # Run infinte loop.
    while True:
        # Read frame.
        grabbed, frame = cap.read()
        # Break if no frame.
        if grabbed is False:
            break
        # Resize frame.
        frame = cv2.resize(frame, (720, 480))
        # Calculate fps.
        frames += 1
        curr_time = time.time() - start
        fps = round(frames/curr_time)
        # Display image with FPS.
        cv2.putText(frame, 'FPS: {}'.format(fps),
                    (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Multi-Threaded", frame)
        # Display frame for 1 ms and break if user presses 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Run for 60 seconds if benchmark.
        if args.benchmark and curr_time > 60:
            break

    # Print metrics.
    print("Total Frames: {}".format(frames))
    print("Time Taken: {:.2f}".format(time.time()-start))
    print("FPS: {}".format(fps))
    # Clean up.
    cap.release()
    cv2.destroyAllWindows()
