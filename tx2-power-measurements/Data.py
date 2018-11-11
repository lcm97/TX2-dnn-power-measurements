import numpy as np
from keras.utils import to_categorical


def get_static_frame_and_stacked_opt_flows():
    static_frames = []
    opt_flow_stacks = []

    for i in range(16):
        static_frame = np.random.rand(12, 16, 3)
        static_frames.append(static_frame)

        opt_flow_stack = np.random.rand(12, 16, 20)
        opt_flow_stacks.append(opt_flow_stack)

    return np.array(static_frames), np.array(opt_flow_stacks)


def get_class_one_hot():
    y = to_categorical(np.random.randint(256, size=(16, 1)), num_classes=256)
    return y


def validation_generator():
    """Return a generator of optical frame stacks that we can use to test."""
    while 1:
        X_spatial_batch = []
        X_temporal_batch = []
        y_batch = []

        for row in range(10):
            # Get the stacked optical flows from disk.
            X_spatial, X_temporal = get_static_frame_and_stacked_opt_flows()

            # Get the corresponding labels
            y = get_class_one_hot()

            X_spatial_batch.append(X_spatial)
            X_temporal_batch.append(X_temporal)
            y_batch.append(y)

        X_batch = [np.array(X_spatial_batch), np.array(X_temporal_batch)]
        y_batch = np.array(y_batch)

        yield X_batch, y_batch
