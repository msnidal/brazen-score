import subprocess

# Neural network
# Following (horizontal, vertical) coordinates
WINDOW_PATCH_SHAPE = (8, 8)
PATCH_DIM = 8
ENCODER_EMBEDDING_DIM = 96
DECODER_EMBEDDING_DIM = 2048
NUM_HEADS = 8
FEED_FORWARD_EXPANSION = 4  # Expansion factor for self attention feed-forward
DECODER_FEED_FORWARD_EXPANSION = 2  # Expansion factor for self attention feed-forward
ENCODER_BLOCK_STAGES = (4, 4)  # Number of transformer blocks in each of the 4 stages
NUM_DECODER_BLOCKS = 2  # Number of decoder blocks
REDUCE_FACTOR = 8  # reduce factor (increase in patch size) in patch merging layer per stage

# Dataset
LABEL_MODE = "semantic"
NUM_SYMBOLS = 758 if LABEL_MODE == "agnostic" else 1781
LABEL_LENGTH = 75 if LABEL_MODE == "agnostic" else 58

RAW_IMAGE_SHAPE = (2048, 2048)
IMAGE_SHAPE = (512, 512)  # rough ratio that's easily dividible

# Training
BATCH_SIZE = 24
OPTIMIZE_EVERY = 8
EPOCH_SIZE = 1
SAVE_EVERY = 1000

LEARNING_RATE = 3e-4
BETAS = (0.9, 0.95)
EPS = 1e-9
WEIGHT_DECAY = 0.1

NUM_WORKERS = 8

DROPOUT_RATE = 0.05
GRAD_NORM_CLIP = 1.0

try:
    GIT_COMMIT = subprocess.check_output(["git", "describe", "--always"]).strip()
except subprocess.CalledProcessError as e:
    print(e.output)
    GIT_COMMIT = "unknown"

WARMUP_SAMPLES = 300000
EXIT_AFTER = 2100000

STANDARD_DEVIATION = 0.02

IMAGE_MEAN = 0.5
IMAGE_STANDARD_DEVIATION = 0.5

DEFAULT_TOKEN_MAP_LENGTH = 779
DEFAULT_MAX_LABEL_LENGTH = 68

DEFAULT_SEED = 0
DEFAULT_DATASET = "symposium"

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
        decoder_feed_forward_expansion=DECODER_FEED_FORWARD_EXPANSION,
        encoder_block_stages=ENCODER_BLOCK_STAGES,
        num_decoder_blocks=NUM_DECODER_BLOCKS,
        reduce_factor=REDUCE_FACTOR,
        label_length=DEFAULT_MAX_LABEL_LENGTH,
        num_symbols=DEFAULT_TOKEN_MAP_LENGTH,
        batch_size=BATCH_SIZE,
        optimize_every=OPTIMIZE_EVERY,
        save_every=SAVE_EVERY,
        eps=EPS,
        betas=BETAS,
        learning_rate=LEARNING_RATE,
        num_workers=NUM_WORKERS,
        dropout_rate=DROPOUT_RATE,
        weight_decay=WEIGHT_DECAY,
        grad_norm_clip=GRAD_NORM_CLIP,
        git_commit=GIT_COMMIT,
        warmup_samples=WARMUP_SAMPLES,
        exit_after=EXIT_AFTER,
        standard_deviation=STANDARD_DEVIATION,
        image_mean=IMAGE_MEAN,
        image_standard_deviation=IMAGE_STANDARD_DEVIATION,
        seed=DEFAULT_SEED,
        model_loaded=None,
        dataset=DEFAULT_DATASET
    ):
        params = locals()
        for param in params:
            setattr(self, param, params[param])

        self.set_dataset_properties(num_symbols, label_length)
        
    def set_dataset_properties(self, num_symbols, label_length):
        """ We have different number of symbols and label length based on the dataset properties
        """
        self.num_symbols = num_symbols

        self.beginning_of_sequence = num_symbols
        self.end_of_sequence = num_symbols + 1
        self.padding_symbol = num_symbols + 2
        self.total_symbols = num_symbols + 3

        self.label_length = label_length
        self.sequence_length = label_length + 1 # includes EOS symbol