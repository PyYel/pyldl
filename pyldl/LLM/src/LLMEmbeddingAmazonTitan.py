import os, sys

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from accelerate import init_empty_weights
import numpy as np
import torch

from .LLM import LLM


class LLMEmbeddingTitanV2(LLM):
    """
    Wrapper for Amazon Titan v2 text embedding model hosted on Hugging Face: `amazon/Titan-text-embeddings-v2`.

    Versions
    --------
    - ``'titan-v2'`` _(default)_ : Amazon Titan v2 embedding model.
        - Initializes with ``'amazon/Titan-text-embeddings-v2'`` weights for text embedding.
        - Requires ~1.2GB of RAM/VRAM in full float32 precision.
        - Input truncated when longer than 512 tokens.
        - Returns a **1024-dimensional dense vector**.

    Note
    ----
    - Multiple tasks may be supported. See ``load_model()``.
    - Quantization is not supported (casting from float32 causes instability).
    """
    def __init__(self, weights_path: str = None, version: str = "titan-v2") -> None:
        """
        Initializes the Titan v2 embedding model.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the model weights should be saved. If None, the current working
            directory path will be used instead.
        """
    
        self.version = version
        if version == "titan-v2": 
            super().__init__(model_name="amazon/Titan-text-embeddings-v2", weights_path=weights_path)
        else:
            print("LLMEmbeddingSentenceTransformer >> Warning: Invalid model version, model 'titan-v2' will be used instead.")
            self.version = "titan-v2"
            super().__init__(model_name="amazon/Titan-text-embeddings-v2", weights_path=weights_path)

        return None


    def load_model(self, display: bool = False):
        """
        Loads the Titan v2 model.

        Parameters
        ----------
        display: bool, False
            Prints the model's device mapping if ``True``.

        Note
        ----
        - Quantization is not available.
            - Reason: Casting from float32 models results in unstable behavior.
        """
        with init_empty_weights(include_buffers=True):
            empty_model = AutoModel.from_pretrained(self.model_folder)
            self._device_map(model=empty_model, dtype_correction=1, display=display)
            del empty_model

        # MODEL SETUP (loading)
        self.model = AutoModel.from_pretrained(
            self.model_folder,
            trust_remote_code=True,
            device_map=self.device_map
        )

        # TOKENIZER SETUP (loading)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_folder,
            clean_up_tokenization_spaces=True,
            padding=True,
            truncation=True
        )

        if display:
            print(f"Model loaded on device: {self.device_map}")

        # PIPELINE SETUP (init)
        self.pipe = self._pipeline
        return None


    def _pipeline(self, prompt: str):
        """
        Custom pipeline object for Titan embeddings.
        """
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        encoded_input = self.tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, attention_mask=encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings


    def evaluate_model(self, 
                      prompts: list[str], 
                      display: bool = False,
                      **kwargs) -> list[list[float]]:
        """
        Embeds a batch of inputs with Titan v2.

        Parameters
        ----------
        prompts: list[str]
            The list of prompts to embed.
        display: bool
            Whether to print the model output. Default is False.

        Returns
        -------
        embedding_results: list[list[float]]
            The embedding results as a list of flattened vectors. Order matches input.
        """
        prompts = self._preprocess(prompts=prompts)

        results = []
        for prompt in prompts:
            results.append(self.pipe(prompt))

        return self._postprocess(results=results, display=display)


    def _preprocess(self, prompts: list[str], **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, list):
            print(f"LLMEmbeddingTitanV2 >> Error: Model's input should be of type 'list[str]', got '{type(prompts)}' instead.")
        return prompts


    def _postprocess(self, results: list[list], display: bool = False, **kwargs):
        embedding_results = [0] * len(results)
        for idx, result in enumerate(results):
            embedding_results[idx] = np.array(result).flatten().tolist()

        if display:
            print(*embedding_results, sep="\n")
        return embedding_results