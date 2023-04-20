import sys
import os
import torch

from model import Whisper, ModelDimensions

print("Using PyTorch version: {}\n".format(torch.__version__))

if len(sys.argv) <= 1:
	print("Usage: python export-whisper-onnx.py [whisper-model-name]")
	exit(0)

modelName = sys.argv[1]

validModelNames = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v1", "large-v2"]
	
if not modelName in validModelNames:
	print("Error: model name must be one of {}".format(", ".join(validModelNames)))
	exit(1)

checkpoint = torch.load("pytorch-models/{}.pt".format(modelName), map_location=torch.device('cpu'))

modelDims = ModelDimensions(**checkpoint["dims"])
whisper = Whisper(modelDims, modelName)
whisper.load_state_dict(checkpoint["model_state_dict"])

whisper = whisper.to("cpu")

batchSize = 1

audioEncoder = whisper.encoder

audioEncoderRandomInputs = torch.randn(batchSize, modelDims.n_mels, modelDims.n_audio_ctx * 2)

encodedFeatures = whisper.encoder(audioEncoderRandomInputs)

outputDir = "onnx-models/{}".format(modelName)

os.makedirs(outputDir, exist_ok=True)

torch.onnx.export(audioEncoder, audioEncoderRandomInputs, "{}/encoder.onnx".format(outputDir), input_names=["mel"], output_names=["output"], opset_version=14, verbose=True)

textDecoder = whisper.decoder

tokens = torch.tensor([[0]], dtype=torch.int64)
kvCache = torch.from_numpy(whisper.new_kv_cache(batchSize, 1))
offset = 0

torch.onnx.export(textDecoder, (tokens, encodedFeatures, kvCache, offset), "{}/decoder.onnx".format(outputDir), input_names=["tokens", "audio_features", "kv_cache", "offset"], output_names=["logits", "output_kv_cache", "cross_attention_qks"],
dynamic_axes={
	"tokens": [0, 1],
	"audio_features": [0],
	"kv_cache": [1, 2],
	"output_kv_cache": [1, 2],
	"cross_attention_qks": [1, 3, 4],
}, opset_version=14, verbose=False)
