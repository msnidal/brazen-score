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
LABEL_LENGTH = 75 if LABEL_MODE == "agnostic" else 58

RAW_IMAGE_SHAPE = (2048, 2048)
IMAGE_SHAPE = (1024, 1024)  # rough ratio that's easily dividible

# Training
BATCH_SIZE = 16
EPOCH_SIZE = 1

LEARNING_RATE = 1e-3
BETAS = (0.9, 0.98)
EPS = 1e-9

NUM_WORKERS = 4


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
        label_length=LABEL_LENGTH,
        num_symbols=NUM_SYMBOLS,
        batch_size=BATCH_SIZE,
        eps=EPS,
        betas=BETAS,
        learning_rate=LEARNING_RATE,
        num_workers=NUM_WORKERS
    ):
        params = locals()
        self.params = {}
        for param in params:
            self.params[param] = params[param]
            setattr(self, param, params[param])

        self.set_dataset_properties(num_symbols, label_length)
        
    def set_dataset_properties(self, num_symbols, label_length):
        """ We have different number of symbols and label length based on the dataset properties
        """
        self.num_symbols = num_symbols
        self.params["num_symbols"] = num_symbols

        self.beginning_of_sequence = num_symbols
        self.end_of_sequence = num_symbols + 1
        self.padding_symbol = num_symbols + 2
        self.total_symbols = num_symbols + 3

        self.label_length = label_length
        self.params["label_length"] = label_length

        self.total_length = label_length + 1 # includes EOS symbol
    
    def __eq__(self, other):
        return self.params == other.params
