"""
Script to run and display video stream using OpenCV.
Parallel implementation using multiprocessing.

Author: Tejas Pandey
"""

import cv2
import time
from multiprocessing import set_start_method
from multiprocessing import Pool, Value, Queue
import argparse


class VideoCapture:
    """
    Video Capture class adapted for multiprocessing.
    """
    
    def __init__(self, src):
        """
        Initialize camera capture from src.
        
        Arguments:
            src {string|int} -- Path to src | Camera
        """

        self.src = src        
        self.cap = cv2.VideoCapture(src)
        assert self.cap.isOpened()
    
    def __getstate__(self):
        """
        Returns current state.
        
        Returns:
            self.src -- Returns existing camera source.
        """
        self.cap.release()
        return self.src

    def __setstate__(self, state):
        """
        Sets current state.
        
        Arguments:
            state {string|int} -- Existing camera source.
        """
        self.src = state
        self.cap = cv2.VideoCapture(self.src)
        assert self.cap.grab(), "Could not initialize capture from child process."

    def read(self):
        """
        Grab frame from video capture.
        
        Raises:
            Exception: [description]
        
        Returns:
            grabbed {bool} -- Whether capture was successful.
            frame {numpy array} -- Current frame.
        """
        
        return self.cap.read()

    def release(self):
        """
        Release memory.
        """

        self.cap.release()

    
def stream(cap, Q, running):
    """
    Push frames to queue while running.
    """

    while running.value:
        grabbed, frame = cap.read()
        # Break if no frame.
        if not grabbed:
            break
        # Resize frame and pass to Queue.
        frame = cv2.resize(frame, (720, 480))
        Q.put(frame)

    # Finish and release capture object.
    cap.release()


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=0)
    parser.add_argument("--benchmark", action='store_true')
    parser.add_argument("--num_cores", default=2)
    args = parser.parse_args()

    # Initialize camera and metrics.
    cap = VideoCapture(args.src)
    frames = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()
    running = Value('i', 1)
    Q = Queue()

    # Start child process.
    pool = Pool(args.num_cores, stream, (cap, Q, running))

    # Run infinte loop.
    while True:
        # Read frame.
        frame = Q.get()
        # Calculate fps.
        frames += 1
        curr_time = time.time() - start
        fps = round(frames/curr_time)

        # Display image with FPS.
        cv2.putText(frame, 'FPS: {}'.format(fps),
                    (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Multi-Processing", frame)

        # Display frame for 1 ms and break if user presses 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running.value = False
            break
        # Run for 60 seconds if benchmark.
        if args.benchmark and curr_time > 60:
            running.value = False
            break

    # Print metrics.
    print("Total Frames: {}".format(frames))
    print("Time Taken: {:.2f}".format(time.time()-start))
    print("FPS: {}".format(fps))

    # Clean up.
    pool.terminate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    set_start_method("spawn")
    main()
