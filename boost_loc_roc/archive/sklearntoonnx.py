"""Script to load a pre-trained sklearn model, convert it to ONNX format and
save it. Archive.

This script was used to convert the voting model used in the LoC-RoC project
to ONNX format. It should be run with the same environment as the one used to
train the model, specifically with version 1.0.2 of the sklearn library,
version 1.6.0 or mne and version 1.16.0 of skl2onnx.
The model was converted to ONNX format in order to not be dependent on the
sklearn library version it was trained with.
"""
import mne
from onnxruntime import InferenceSession
import numpy as np
import pandas as pd
from boost_loc_roc.archive.model import load_voting_skmodel
from eeg_features import compute_input_sample
from utils.preprocessing import truncate_fif
from skl2onnx import to_onnx


def get_input_sample(
    raw,
    epochs_duration=30,
    shift=30,
    n_fft=512,
    n_overlap=128,
    num_features=50
):
    """Computes the first input sample for the voting model.

    Parameters
    ----------
    raw: mne.io.Raw
        Raw EEG data.
    epochs_duration, shift, n_fft, n_overlap, num_features: int
        Parameters passed to `compute_input_sample`.

    Returns
    -------
    np.array, shape (1, num_features)"""
    input_samples = compute_input_sample(
        raw,
        epochs_duration,
        shift,
        n_fft,
        n_overlap,
        num_features
    )
    first_sample = input_samples.iloc[0, :].to_numpy(np.float32).reshape(1, -1)
    return first_sample


def convert_voting_onnxmodel(raw, output_path):
    """Loads pretrained sklearn voting model to ONNX format and optionnaly
    saves it.
    Parameters
    ----------
    raw: mne.io.Raw
        Raw EEG data.
    output_path: str or None.
        Path to save the onnx model. If None, the model is not saved.

    Returns
    -------
    onnx_model: onnx model
            Model in onnx format.
    """
    # Load sklearn model
    model = load_voting_skmodel()
    # Change flatten_transform to False to avoid skl2onnx error
    model.set_params(flatten_transform=False)
    # Load one input sample
    input_sample = get_input_sample(raw)
    # Convert sklearn model to onnx
    onnx_model = to_onnx(model, input_sample, target_opset=12)
    if output_path is not None:
        # Save onnx model
        with open(output_path + "voting_model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
    return onnx_model


# Load data
print('loading data...')
raw = mne.io.read_raw_fif("data/raw/example_eeg.fif")
raw = truncate_fif(raw)
input_samples = compute_input_sample(
    raw,
    epochs_duration=30,
    shift=30,
    n_fft=512,
    n_overlap=128,
    num_features=50
)
first_input_sample = get_input_sample(raw)
print(('Done.'))
# Convert onnx model and make prediction
print('Converting onnx model and saving it and making prediction...')
onnx_model = convert_voting_onnxmodel(raw, 'boost_loc_roc/model_weights/')
session = InferenceSession(
    onnx_model.SerializeToString(),
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
pred_onnx = session.run(
    None,
    {input_name: input_samples.to_numpy().astype(np.float32)}
)
print('Done.')
# Load onnx model and make predictions
print('Loading onnx model and making predictions...')
sessionloaded = InferenceSession(
    "boost_loc_roc/model_weights/voting_model.onnx",
    providers=["CPUExecutionProvider"]
)
pred_loaded = sessionloaded.run(
    None,
    {input_name: input_samples.to_numpy().astype(np.float32)}
)
print('Done.')
# Load sklearn model and make prediction
sk_model = load_voting_skmodel()
pred_sklearn = sk_model.predict_proba(input_samples)

# Compare predictions
df_predproba = pd.DataFrame(pred_onnx[1])
df_predprobaloaded = pd.DataFrame(pred_loaded[1])
print(
    'Are predictions equal?',
    np.allclose(df_predproba, pred_sklearn, rtol=1e-03, atol=1e-05),
    np.allclose(df_predprobaloaded, pred_sklearn, rtol=1e-03, atol=1e-05)
)
