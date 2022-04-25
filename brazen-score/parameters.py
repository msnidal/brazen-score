# Neural network
# Following (horizontal, vertical) coordinates
WINDOW_PATCH_SHAPE = (8, 8)
PATCH_DIM = 8
ENCODER_EMBEDDING_DIM = 128
DECODER_EMBEDDING_DIM = 1024
NUM_HEADS = 4
FEED_FORWARD_EXPANSION = 2  # Expansion factor for self attention feed-forward
ENCODER_BLOCK_STAGES = (2, 2)  # Number of transformer blocks in each of the 4 stages
NUM_DECODER_BLOCKS = 1  # Number of decoder blocks
REDUCE_FACTOR = 16  # reduce factor (increase in patch size) in patch merging layer per stage

# Dataset
LABEL_MODE = "semantic"
NUM_SYMBOLS = 758 if LABEL_MODE == "agnostic" else 1781
SEQUENCE_DIM = 75 if LABEL_MODE == "agnostic" else 58

BEGINNING_OF_SEQUENCE = NUM_SYMBOLS
END_OF_SEQUENCE = NUM_SYMBOLS + 1
PADDING_SYMBOL = NUM_SYMBOLS + 2
TOTAL_SYMBOLS = NUM_SYMBOLS + 3

RAW_IMAGE_SHAPE = (2048, 2048)
IMAGE_SHAPE = (1024, 1024)  # rough ratio that's easily dividible

# Training
BATCH_SIZE = 16
EPOCH_SIZE = 1

LEARNING_RATE = 1e-3
BETAS = (0.9, 0.98)
EPS = 1e-9


class BrazenParameters:
    def __init__(
        self,
        window_patch_shape=WINDOW_PATCH_SHAPE,
        image_shape=IMAGE_SHAPE,
        patch_dim=PATCH_DIM,
        encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
        decoder_embedding_dim=DECODER_EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        feed_forward_expansion=FEED_FORWARD_EXPANSION,
        encoder_block_stages=ENCODER_BLOCK_STAGES,
        num_decoder_blocks=NUM_DECODER_BLOCKS,
        reduce_factor=REDUCE_FACTOR,
        output_sequence_dim=SEQUENCE_DIM,
        num_symbols=NUM_SYMBOLS,
        beginning_of_sequence=BEGINNING_OF_SEQUENCE,
        end_of_sequence=END_OF_SEQUENCE,
        padding_symbol=PADDING_SYMBOL,
        total_symbols=TOTAL_SYMBOLS,
        batch_size=BATCH_SIZE,
        eps=EPS,
        betas=BETAS,
        learning_rate=LEARNING_RATE
    ):
        params = locals()
        for param in params:
            setattr(self, param, params[param])
