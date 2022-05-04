class config():  
    VALIDATION_SIZE = 0.1
    BATCH_SIZE = 32
    MAX_EPOCHS = 25
    EMBED_DIM = 66
    DENSE_DIM = 66
    NUM_HEADS = 2
    X_LEN = 32
    Y_LEN = 66
    NUM_ENCODER_TOKENS = 48
    NUM_DECODER_TOKENS = 64
    OPTIMIZER="adam" 
    LOSS="categorical_crossentropy"
    METRICS="accuracy"