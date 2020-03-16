"""
Face detection using Openvino on single thread.

TODO: Only camera is async. Not openvino.

Author: Tejas Pandey
"""

import numpy as np
import cv2
import time
from openvino.inference_engine import IENetwork, IECore
from multi_threaded import AsyncVideoCapture
import argparse


def unsupported_layers(ie, net, device="CPU"):
    """
    Return unsupported layers if any.

    Args:
        ie: IECore object.
        net: IENetwork object.
        device: Device name.
    
    Returns:
        Number of unsupported layers and their keys.
    """

    # Get supported layers.
    supported_layers = ie.query_network(net, device)
    # Get layers not part of supported layers.
    not_supported_layers = [
        l for l in net.layers.keys() if l not in supported_layers]
    # Return number of unsupported layers and their keys.
    return len(not_supported_layers), not_supported_layers


def load_network(model_xml, model_bin, device="CPU"):
    """
    Creates a network from model files and loads them to device.

    Args:
        model_xml: Path to model xml.
        model_bin: Path to model bin.
        device: Device name.

    Returns:
        exec_net: Exectuable network loaded in device.
        input_shape: Expected input shape.
        input_key: Input layer key.
        output_key: Output layer key.
    """

    # Load network from model files.
    net = IENetwork(model_xml, model_bin)
    # Add extenstions for SSD.
    ie = IECore()
    cpu_ext = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.dylib"
    ie.add_extension(cpu_ext, "CPU")
    assert not unsupported_layers(ie, net)[0], "Found unsupported layers!"
    # Get input and output keys.
    input_key = next(iter(net.inputs))
    output_key = next(iter(net.outputs))
    # Get shape expected by input layer.
    input_shape = net.inputs[input_key].shape
    # Load network to device.
    exec_net = ie.load_network(network=net, device_name=device)
    # Return executable net, input shape, input and output keys.
    return exec_net, input_shape, input_key, output_key


def display_output(image, res):
    """
    Decode and display the output.

    Args:
        image: Input image.
        res: Network result.

    Returns:
        image: Image with boxes drawn on it.
    """

    # Initialize boxes and classes.
    boxes, classes = {}, {}
    data = res[0][0]
    # Enumerate over all proposals.
    for number, proposal in enumerate(data):
        # For all proposals with confidence greater than 0.5.
        if proposal[2] > 0.5:
            # Get index.
            imid = np.int(proposal[0])
            # Image height and width.
            ih, iw = image.shape[:-1]
            # Class label.
            label = np.int(proposal[1])
            # Output confidence.
            confidence = proposal[2]
            # Resize box predictions for input image.
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            # Add boxes and classes.
            if not imid in boxes.keys():
                boxes[imid] = []
                boxes[imid].append([xmin, ymin, xmax, ymax])
            if not imid in classes.keys():
                classes[imid] = []
                classes[imid].append(label)

    # Draw boxes for all predictions.
    for imid in classes:
        for box in boxes[imid]:
            cv2.rectangle(image, (box[0], box[1]),
                          (box[2], box[3]), (232, 35, 244), 2)

    # Return image with boxes drawn on it.
    return image


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model xml.")
    parser.add_argument("weights", help="Path to model weights.")
    parser.add_argument("--src", help="Path to src.", default=0)
    args = parser.parse_args()
    
    # Get executable network, input shape, input and output keys.
    exec_net, input_shape, input_key, output_key = load_network(
        args.model, args.weights)

    # Initialize video capture from source and all our metrics.
    cap = AsyncVideoCapture(args.src)
    frames = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()

    # Run infinite loop.
    while True:
        # Read frame.
        grabbed, frame = cap.read()
        # Break if no frame.
        if grabbed is False:
            break

        # Resize frame to what's expected by network.
        input_image = cv2.resize(frame, (input_shape[3], input_shape[2]))
        # Add axis for batch size.
        input_image = np.expand_dims(input_image, 0)
        # Change NHWC -> NCHW.
        input_image = np.transpose(input_image, [0, 3, 1, 2])
        # Run inference.
        res = exec_net.infer(inputs={input_key: input_image})
        # Get output from last layer.
        res = res[output_key]
        # Decode output results on image.
        output = display_output(frame, res)

        # Calculate fps.
        frames += 1
        curr_time = time.time() - start
        fps = round(frames/curr_time)

        # Display image with fps.
        cv2.putText(frame, 'FPS: {}'.format(fps),
                    (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Default", frame)

        # Display frame for 1 ms and break if user presses 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up.
    cap.release()
    cv2.destroyAllWindows()
