"""
Script to run video stream using OpenCV on a single thread.

Author: Tejas Pandey
"""

import cv2
import time
import argparse


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=0,
                        help="Path to src file. Defaults to camera.")
    parser.add_argument("--benchmark", action='store_true',
                        help="Run for 60 seconds.")
    args = parser.parse_args()

    # Initialize camera and metrics.
    cap = cv2.VideoCapture(args.src)
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
        cv2.imshow("Default", frame)
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
