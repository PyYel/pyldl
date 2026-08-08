import os, sys

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline
from accelerate import init_empty_weights
import numpy as np

from .LLM import LLM


class LLMEmbeddingUAE(LLM):
    """
    A collection of pretrained models based on the WhereIsAI Universal Angle Embedding backbone, fine-tuned for text embedding.
    """
    def __init__(self, weights_path: str = None, version: str = "large") -> None:
        """
        Initializes a pretrained model based on the WhereIsAI Universal Angle Embedding backbone backbone, fine-tuned for text embedding.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'large'`` _(default)_ : The base 335 million parameters version of UAE. 
            - Initializes the model with ``'WhereIsAI/UAE-Large-V1'`` weights for text embedding.
            - For the full float32 model, requires 0.5Go of RAM/VRAM. 

        Note
        ----
        - Multiple tasks may be supported. See ``load_model()``.
        - Quantization isn't supported. See ``load_model()``.
        """

        self.version = version
        if version == "large": 
            super().__init__(model_name="WhereIsAI/UAE-Large-V1", weights_path=weights_path)
        else:
            print("LLMEmbeddingUAE >> Warning: Invalid model version, model 'large' will be used instead.")
            self.version = "base"
            super().__init__(model_name="WhereIsAI/UAE-Large-V1", weights_path=weights_path)

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


        # PIPELINE SETUP (init)
        self.pipe = pipeline(task='feature-extraction', 
                             model=self.model,
                             tokenizer=self.tokenizer)
        return None

    
    def evaluate_model(self, 
                      prompts: list[str], 
                      display: bool = False,
                      vect_length: int = 1024,
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

        return self._postprocess(results=results, vect_length=vect_length, display=display)


    def _preprocess(self, prompts: list[str], **kwargs):
        """
        Preprocesses the pipeline inputs.
        """

        if isinstance(prompts, str): prompts = [prompts]
        if not isinstance(prompts, list): 
            print(f"LLMEmbeddingUAE >> Error: Model's input should be of type 'list[str]', got '{type(prompts)}' instead.")

        return prompts


    def _postprocess(self, results: list[list], vect_length: int, display: bool = False, **kwargs):
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
            embedding_results[idx] = np.array(result).flatten().tolist()[0:vect_length]

        if display: print(*embedding_results, sep="\n")

        return embedding_results

