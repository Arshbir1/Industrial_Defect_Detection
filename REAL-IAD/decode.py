from SemanticDecoder import secure_decode, interpret

data = secure_decode("mvtec_secure_output/000.png.bin")
interpret(data)