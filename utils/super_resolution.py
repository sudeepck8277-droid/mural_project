import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(
    package="MuralRestoration"
)
class PixelShuffle(layers.Layer):

    def __init__(
        self,
        scale=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = int(scale)

    def call(self, inputs):
        return tf.nn.depth_to_space(
            inputs,
            self.scale
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale
        })
        return config


class MuralSuperResolver:

    def __init__(
        self,
        model_path,
        max_input_dimension=None
    ):
        if max_input_dimension is None:
            # A 512px EDSR input requires several GB of temporary memory and
            # takes many minutes on CPU. Keep the web app responsive with a
            # balanced default, while allowing full-resolution inference on a
            # capable machine through MURAL_SR_MAX_INPUT_DIMENSION=512.
            default_dimension = (
                512
                if tf.config.list_physical_devices("GPU")
                else 256
            )
            max_input_dimension = os.environ.get(
                "MURAL_SR_MAX_INPUT_DIMENSION",
                default_dimension
            )

        self.max_input_dimension = (
            max(32, int(max_input_dimension))
        )

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "PixelShuffle": PixelShuffle
            },
            compile=False,
            safe_mode=False
        )

        print(
            "Super-resolution model loaded:",
            self.model.name,
            f"(maximum input dimension: {self.max_input_dimension}px)"
        )

    def prepare_input(
        self,
        image_rgb
    ):
        h, w = image_rgb.shape[:2]

        max_dim = max(h, w)

        if max_dim <= self.max_input_dimension:
            return image_rgb

        scale = (
            self.max_input_dimension
            /
            max_dim
        )

        new_w = max(
            1,
            int(round(w * scale))
        )

        new_h = max(
            1,
            int(round(h * scale))
        )

        return cv2.resize(
            image_rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    def upscale(
        self,
        image_rgb
    ):
        prepared = self.prepare_input(
            image_rgb
        )

        x = (
            prepared.astype(
                np.float32
            )
            /
            255.0
        )

        # Calling the model directly avoids the extra dataset/progress
        # machinery created by Model.predict for a single image.
        prediction = self.model(
            x[None, ...],
            training=False
        ).numpy()[0]

        prediction = np.clip(
            prediction,
            0.0,
            1.0
        )

        return (
            prediction * 255.0
        ).astype(
            np.uint8
        )
