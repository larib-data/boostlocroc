"""Functions to handle ONNX models.

- onx_make_session: Make an onnx session from a model path to run it.
- onx_make_prediction: Make a prediction from an onnx session and an input
sample.
- onx_predict_class: Assigns a class to an input sample using an onnx session.
- onx_predict_proba: Predict probabilities using an onnx session.
"""

import numpy as np
import pandas as pd
from onnxruntime import InferenceSession


def onx_make_session(path):
    """Make an onnx session from a model path to run it.

    Parameters
    ----------
    path: str
        Path to the onnx model.

    Returns
    -------
    session: onnxruntime.InferenceSession
        Session to run the model.
    """
    return InferenceSession(path, providers=["CPUExecutionProvider"])


def onx_make_prediction(session, input_sample):
    """Make a prediction from an onnx session and an input sample.

    Parameters
    ----------
    session: onnxruntime.InferenceSession
        Session to run the model.
    input_sample: np.array, shape (num_samples, num_features)
        Input sample to predict.

    Returns
    -------
    TODO"""
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_sample.astype(np.float32)})


def onx_predict_class(session, input_sample):
    """Assigns a class to an input sample using an onnx session.

    Parameters
    ----------
    session: onnxruntime.InferenceSession
        Session to run the model.
    input_sample: np.array, shape (num_samples, num_features)
        Input sample to predict. Can be a single sample or multiple samples.

    Returns
    -------
    np.array, shape (num_samples,) TODO: CHECK OUTPUT TYPE AND SHAPE
        Predicted class.
    """
    return onx_make_prediction(session, input_sample)[0]


def onx_predict_proba(session, input_sample):
    """Equivalent to sklearn.predict_proba. Assigns a probability to each
    class on input sample using an onnx session.

    Parameters
    ----------
    session: onnxruntime.InferenceSession
        Session to run the model.
    input_sample: np.array, shape (num_samples, num_features)
        Input sample to predict. Can be a single sample or multiple samples.

    Returns
    -------
    np.array, shape (num_samples, num_classes)
        Probabilities for each class.
    """
    proba_dict = onx_make_prediction(session, input_sample)[1]
    return pd.DataFrame(proba_dict).to_numpy()
