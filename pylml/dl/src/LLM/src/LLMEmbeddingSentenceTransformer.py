import os, sys

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline
from accelerate import init_empty_weights
import numpy as np
import torch

from .LLM import LLM


class LLMEmbeddingSentenceTransformer(LLM):
    """
    A collection of pretrained models based on the Sentence Transformer library templates, fine-tuned for text embedding.
    """
    def __init__(self, weights_path: str = None, version: str = "minilm-l6") -> None:
        """
        Initializes a pretrained model based on the Sentence Transformer library templates, fine-tuned for text embedding.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'minilm-l6'`` _(default)_ : The base 22.7 million parameters version of UAE. 
            - Initializes the model with ``'sentence-transformers/all-MiniLM-L6-v2'`` weights for text embedding.
            - For the full float32 model, requires 0.5Go of RAM/VRAM. 
            - Input trunctaded when longer than 256 tokens.
            - Returns a 384 dimensions dense vector.

        - ``'minilm-l12'`` _(default)_ : The base 118 million parameters version of MiniLLM. 
            - Initializes the model with ``'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'`` weights for text embedding.
            - For the full float32 model, requires 0.5Go of RAM/VRAM. 
            - BERT backbone.
            - Multilingual model.
            - Returns a 384 dimensions dense vector.

        - ``'mpnet'`` _(default)_ : The base 109 million parameters version of UAE. 
            - Initializes the model with ``'sentence-transformers/all-mpnet-base-v2'`` weights for text embedding.
            - For the full float32 model, requires 0.5Go of RAM/VRAM. 
            - Input trunctaded when longer that 384 tokens.
            - Returns a 768 dimensions dense vector.

        Note
        ----
        - Multiple tasks may be supported. See ``load_model()``.
        - Quantization isn't supported. See ``load_model()``.
        """

        self.version = version
        if version == "minilm-l6": 
            super().__init__(model_name="sentence-transformers/all-MiniLM-L6-v2", weights_path=weights_path)
        elif version == "minilm-l12": 
            super().__init__(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", weights_path=weights_path)
        elif version == "mpnet": 
            super().__init__(model_name="sentence-transformers/all-mpnet-base-v2", weights_path=weights_path)
        else:
            print("LLMEmbeddingSentenceTransformer >> Warning: Invalid model version, model 'minilm-l6' will be used instead.")
            self.version = "minilm-l6"
            super().__init__(model_name="sentence-transformers/all-MiniLM-L6-v2", weights_path=weights_path)

        return None
    

    def load_model(self, display: bool = False):
        """
        Loads the selected model.

        Parameters
        ----------
        display: bool, False
            Prints the model's device mapping if ``True``.

        Note
        ----
        - Quantization is not available.
            - Reason: Casting from these float32 models results in highly unstable models.
        """

        with init_empty_weights(include_buffers=True):
            empty_model = AutoModel.from_pretrained(self.model_folder)
            self._device_map(model=empty_model, dtype_correction=1, display=display)
            del empty_model

        # MODEL SETUP (loading)
        self.model = AutoModel.from_pretrained(self.model_folder, 
                                               trust_remote_code=True, 
                                               device_map=self.device_map)

        # TOKENIZER SETUP (loading)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder, 
                                                       clean_up_tokenization_spaces=True, padding=True, truncation=True)

        if display:
            print(f"Model loaded on device: {self.device_map}")


        # PIPELINE SETUP (init) # The Transformer library does not offer pipelines for this task.
        self.pipe = self._pipeline

        return None


    def _pipeline(self, prompt: str):
        """
        Custom pipeline object.
        """
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
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
                      **kwargs) -> list[dict[str, float]]:
        """
        Embedds an input.

        Parameters
        ----------
        prompts: list[str]
            The list of prompts to classify.
        display: bool
            Whether to print the model output. Default is True.

        Returns
        -------
        embedding_results: list[list[float]]
            The embedding results as a list of sorted flattened vectors. The output is in the same order as the input.
        """

        prompts = self._preprocess(prompts=prompts)

        results = []
        for prompt in prompts:
            results.append(self.pipe(prompt))

        return self._postprocess(results=results, display=display)


    def _preprocess(self, prompts: list[str], **kwargs):
        """
        Preprocesses the pipeline inputs.
        """

        if isinstance(prompts, str): prompts = [prompts]
        if not isinstance(prompts, list): 
            print(f"LLMEmbeddingSentenceTransformer >> Error: Model's input should be of type 'list[str]', got '{type(prompts)}' instead.")

        return prompts


    def _postprocess(self, results: list[list], display: bool = False, **kwargs):
        """
        Postprocesses the pipeline output to make it directly readable.

        Parameters
        ----------
        results: list[dict]
            The pipeline outputs that will be cherry-picked to only return the predicted label or output.
        display: bool
            Whereas to display the processing bar or not.
        """

        embedding_results = [0] * len(results)
        for idx, result in enumerate(results):
            embedding_results[idx] = np.array(result).flatten().tolist()
            
        if display: print(*embedding_results, sep="\n")

        return embedding_results

