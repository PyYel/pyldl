import os, sys

from pylml.dl import *

# This is a test file. Run it with pyyel as an installed package.

for cls in [
    CNNDetectionFasterRCNN,
    CNNDetectionSSD,
    LLMEmbeddingSentenceTransformer,
    LLMEmbeddingUAE,
    LLMInstructGenerationLlama3,
    LLMInstructGenerationMistral,
]:
    try:
        cls()
    except Exception as e:
        print(e)
