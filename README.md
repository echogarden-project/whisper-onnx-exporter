# Whisper ONNX models

OpenAI Whisper speech recognition models, exported to ONNX.

The original source code has been simplified to only the core model file (`model.py`).

Taking the code in [`whisper-openvino`](https://github.com/zhuzilin/whisper-openvino) as a starting point, the model's key-value structure has been modified to pass as an input and output, with no need for hooks.

The `TextDecoder`, `ResidualAttentionBlock` and `MultiHeadAttention` classes have also been modified to directly output the cross-attention weights, without any hooks.

The exported ONNX model is primarily designed to be used from Echogarden, which has its own implementation of the higher-level Whisper API (other than the core model), and is written in JavaScript. The code doesn't include a way to use the exported models from Python. However, since it is closely related to the code on [`whisper-openvino`](https://github.com/zhuzilin/whisper-openvino), which does work from Python, it should be possible to make it work with it, with some modifications.

## Usage

Ensure you have PyTorch installed.

Copy the official Whisper model files (`.pt`) to the `pytorch-models` subdirectory.

To get the models you can use the official Whisper CLI, which would auto-download a model as needed. On Windows, the downloaded models should be stored at `%userprofile%\.cache\whisper`.

Run:
```
python export-whisper-onnx.py [whisper-model-name]
```

For example:
```
python export-whisper-onnx.py tiny
```

The exported encoder and decoder ONNX models would be located at:
```
onnx-models/tiny/encoder.onnx
onnx-models/tiny/decoder.onnx
```

## License

MIT
