import onnx

import onnxruntime as ort

modelPath = "onnx-models/large-v3-turbo/encoder_fp16.onnx"

result = onnx.checker.check_model(modelPath, full_check = True)

print(result)

#model = onnx.load_model(modelPath, load_external_data=True)

#print("OK")

ort_sess = ort.InferenceSession(modelPath)

print("OK 2")
