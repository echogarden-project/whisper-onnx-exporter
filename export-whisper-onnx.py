import sys
import os
import torch
import shutil

from timeit import default_timer as timer
from model import Whisper, ModelDimensions

torch.manual_seed(0)

def start(): 
	print("Using PyTorch version: {}\n".format(torch.__version__))

	if len(sys.argv) <= 1:
		print("Usage: python export-whisper-onnx.py whisper-model-name [--export-fp16] [--export-fp16-mixed]")
		exit(0)

	modelName = sys.argv[1]

	validModelNames = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]
		
	if not modelName in validModelNames:
		print("Error: model name must be one of {}".format(", ".join(validModelNames)))
		exit(1)
		
	exportFP16 = '--export-fp16' in sys.argv
	exportFP16Mixed = '--export-fp16-mixed' in sys.argv
		
	startTime = timer()
	
	print("")
	print("Loading PyTorch model..")

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
	
	shutil.rmtree(outputDir, ignore_errors = True)
	os.makedirs(outputDir, exist_ok=True)

	outputEncoderFilename = "{}/encoder.onnx".format(outputDir)
	outputDecoderFilename = "{}/decoder.onnx".format(outputDir)

	opset_version = 17
	
	print("")
	print("Exporting audio encoder..")
	
	torch.onnx.export(
		audioEncoder, 
		audioEncoderRandomInputs, 
		outputEncoderFilename, 
		input_names=["mel"], 
		output_names=["output"], 
		opset_version=opset_version, 
		verbose=True)

	print("Exporting text decoder..")
	
	textDecoder = whisper.decoder

	tokens = torch.tensor([[0]], dtype=torch.int64)
	kvCache = torch.from_numpy(whisper.new_kv_cache(batchSize, 1))
	offset = 0
	
	audioDecoderInputs = (tokens, encodedFeatures, kvCache, offset)
	
	torch.onnx.export(
		textDecoder, 
		audioDecoderInputs, 
		outputDecoderFilename, 
		input_names=["tokens", "audio_features", "kv_cache", "offset"], 
		output_names=["logits", "output_kv_cache", "cross_attention_qks"],
		dynamic_axes={
			"tokens": [0, 1],
			"audio_features": [0],
			"kv_cache": [1, 2],
			"output_kv_cache": [1, 2],
			"cross_attention_qks": [1, 3, 4],
		},
		opset_version=opset_version,
		verbose=True)
	
	if exportFP16:
		outputFP16EncoderFilename = "{}/encoder_fp16.onnx".format(outputDir)
		outputFP16DecoderFilename = "{}/decoder_fp16.onnx".format(outputDir)		
		
		print("")
		print("Quantizing encoder to 16-bits..")
		
		import onnx
		from onnxconverter_common import float16
		
		onnx.checker.check_model(outputEncoderFilename, full_check = True)
		
		encoderOnnx = onnx.load(outputEncoderFilename)
		encoderOnnx_fp16 = float16.convert_float_to_float16(encoderOnnx, keep_io_types=True, disable_shape_infer=True)
		
		onnx.save(encoderOnnx_fp16, outputFP16EncoderFilename)

		print("")
		print("Quantizing decoder to 16-bits..")
		
		onnx.checker.check_model(outputDecoderFilename, full_check = True)
		
		decoderOnnx = onnx.load(outputDecoderFilename)
		decoderOnnx_fp16 = float16.convert_float_to_float16(decoderOnnx, keep_io_types=True, disable_shape_infer=True)
		
		onnx.save(decoderOnnx_fp16, outputFP16DecoderFilename)
	
	if exportFP16Mixed:
		outputMixedPrecisionEncoderFilename = "{}/encoder_fp16_mixed.onnx".format(outputDir)
		outputMixedPrecisionDecoderFilename = "{}/decoder_fp16_mixed.onnx".format(outputDir)	
				
		import onnx
		from onnxconverter_common import auto_convert_mixed_precision
		import numpy as np
		
		def validate(res1, res2):
			for r1, r2 in zip(res1, res2):
				if not np.allclose(r1, r2, rtol=0.01, atol=0.1):
					return False
			
			return True
		
		print("")
		print("Quantizing encoder to 16-bits mixed-precision..")		
		
		encoderOnnx = onnx.load("{}/encoder.onnx".format(outputDir))
		
		encoderOnnx_fp16_mixed = auto_convert_mixed_precision(
			encoderOnnx, 
			{ 
				'mel': audioEncoderRandomInputs.detach().numpy() 
			}, 
			validate_fn=validate,
			keep_io_types=True)
		
		onnx.save(encoderOnnx_fp16_mixed, outputMixedPrecisionEncoderFilename.format(outputDir))
		
		print("")
		print("Quantizing decoder to 16-bits mixed-precision..")		

		decoderOnnx = onnx.load("{}/decoder.onnx".format(outputDir))
		decoderOnnx_fp16_mixed = auto_convert_mixed_precision(
			decoderOnnx, 
			{ 
				'tokens': tokens.detach().numpy(), 
				'audio_features': encodedFeatures.detach().numpy(), 
				'kv_cache': kvCache.detach().numpy(), 
				'offset': np.array((offset)).astype(np.int64)
			}, 
			validate_fn=validate, 
			keep_io_types=True)
		
		onnx.save(decoderOnnx_fp16_mixed, outputMixedPrecisionDecoderFilename.format(outputDir))
	
	print("")
	print("\nTotal export time: {:.2f}s".format(timer() - startTime))		

start()
