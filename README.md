# Whisper ONNX exporter

A tool to export OpenAI Whisper speech recognition models to ONNX.

The core model file (`model.py`) has been isolated from the [original Whisper codebase](https://github.com/openai/whisper). Other files are not included or needed.

Taking some of the code in [`whisper-openvino`](https://github.com/zhuzilin/whisper-openvino) as a starting point, the model's key-value structure has been modified to be passed as an input or output, removing the need for hooks.

The `TextDecoder`, `ResidualAttentionBlock` and `MultiHeadAttention` classes have also been further modified to directly output the cross-attention weights, without any hooks.

The exported ONNX models are primarily intended to be used with [Echogarden](https://github.com/echogarden-project/echogarden), which has its own implementation of the higher-level Whisper API, and is written in TypeScript. The code doesn't include a way to use the exported models from Python. However, since it is closely related to the code on [`whisper-openvino`](https://github.com/zhuzilin/whisper-openvino), which adapts the higher-level Python API to use it, it should be possible to make it work with it, with some modifications.

## Downloading pre-exported models

You can download pre-exported models for all sizes, except `large`, `large-v1`, `large-v2`, `large-v3` from the releases section of the [`whisper-onnx-models` repository](https://github.com/echogarden-project/whisper-onnx-models).

## Usage

Ensure you have PyTorch installed.

Copy the official Whisper model files (`.pt`) to the `pytorch-models` subdirectory.

To get the models you can use the official Whisper CLI, which would auto-download a model as needed. On Windows, the downloaded models should be stored at `%userprofile%\.cache\whisper`.

Alternatively, you may find direct download URLs in the [original whisper source code](https://github.com/openai/whisper/blob/e58f28804528831904c3b6f2c0e473f346223433/whisper/__init__.py#L17C1-L17C1).

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
