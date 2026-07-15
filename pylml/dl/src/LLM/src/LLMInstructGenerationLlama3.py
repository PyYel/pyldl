import os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from accelerate import init_empty_weights

from .LLM import LLM


class LLMInstructGenerationLlama3(LLM):
    """
    A collection of pretrained models based on the Meta's Llama3 backbone, fine-tuned for text generation.
    """
    def __init__(self, weights_path: str = None, version: str = "1B") -> None:
        """
        Initializes a pretrained model based on the Meta's Llama3 backbone, fine-tuned for text generation.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'1B'`` _(default)_ : The smallest 1 billion parameters version of Llama 3.2. 
            - Initializes the model with ``'meta-llama/Llama-3.2-1B-Instruct'`` weights for text generation.
            - For the full bfloat16 model, requires 2.5Go of RAM/VRAM. 
        - ``'3B'``: The 3 billion parameters version of Llama 3.2. 
            - Initializes the model with ``'meta-llama/Llama-3.2-3B-Instruct'`` weights for text generation.
            - For the full bfloat16 model, requires ?Go of RAM/VRAM. 
        - ``'8B'``: The 8 billion parameters version of Llama 3.1. 
            - Initializes the model with ``'meta-llama/Llama-3.1-8B-Instruct'`` weights for text generation.
            - For the full bfloat16 model, requires ?Go of RAM/VRAM. 

        Note
        ----
        - Quantization may be supported. See ``load_model()``.
        """

        self.version = version
        if version == "1B": 
            super().__init__(model_name="meta-llama/Llama-3.2-1B-Instruct", weights_path=weights_path)
        elif version == "3B": 
            super().__init__(model_name="meta-llama/Llama-3.2-3B-Instruct", weights_path=weights_path)
        elif version == "8B": 
            super().__init__(model_name="meta-llama/Llama-3.1-8B-Instruct", weights_path=weights_path)
        else:
            print("LLMInstructGenerationLlama3 >> Warning: Invalid model version, model '1B' will be used instead.")
            self.version = "1B"
            super().__init__(model_name="meta-llama/Llama-3.2-1B-Instruct", weights_path=weights_path)

        return None


    def load_model(self, quantization: str = None, display: bool = False):
        """
        Loads the selected model for text generation.

        Parameters
        ----------
        quantization: str, None
            Quantisizes a model to reduce its memory usage and improve speed. Quantization can only be done
            on compatible GPUs, either in 4-bits (``quantization='4b'``) or 8-bits (``quantization='8b'``).  
        display: bool, False
            Prints the model's device mapping if ``True``.

        Note
        ----
        - Make sure you have a combinaison of devices that has enough RAM/VRAM to host the whole model. Extra weights will be sent to CPU RAM, which will
        greatly reduce the computing speed. Additionnal memory needs offloaded to scratch disk (default disk, not recommended).
        - If you lack memory, try quantisizing the models for important performances improvements. This may break some models or lead to more hallucinations.
            - Quantization in 8-bits requires roughly TODO of VRAM
            - Quantization in 4-bits requires roughly TODO of VRAM
        """
        
        if quantization in ["8b", "4b"] and torch.cuda.is_available():
            if quantization == "8b":
                load_in_8bit = True
                load_in_4bit = False
                dtype_correction = 2.0
            if quantization == "4b":
                load_in_8bit = False
                load_in_4bit = True
                dtype_correction = 4.0
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit, 
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,)
                # bnb_4bit_compute_dtype=torch.bfloat16)
            low_cpu_mem_usage = True
        else:
            low_cpu_mem_usage = None
            quantization_config = None
            dtype_correction = 2.0
        
        with init_empty_weights(include_buffers=True):
            empty_model = AutoModelForCausalLM.from_pretrained(self.model_folder, quantization_config=quantization_config)
            self._device_map(model=empty_model, dtype_correction=dtype_correction, display=display)
            del empty_model

        self.model = AutoModelForCausalLM.from_pretrained(self.model_folder, 
                                                          trust_remote_code=True, 
                                                          quantization_config=quantization_config, 
                                                          low_cpu_mem_usage=low_cpu_mem_usage,
                                                          device_map=self.device_map,
                                                        #   attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder, 
                                                       clean_up_tokenization_spaces=True)
        self.pipe = pipeline("text-generation",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             torch_dtype=torch.bfloat16,
                             device_map=self.device_map)
                
        return None

    
    def evaluate_model(self, 
                       prompts: list[str], 
                       contexts: str = "", 
                       max_tokens: int = 1000, 
                       display: bool = False):
        """
        Evaluates a prompt and returns the model answer.

        Parameters
        ----------
        prompt: str
            The model querry.
        context: str
            Enhances a prompt by concatenating a string content beforehand. Default is '', adding the context
            is equivalent to enhancing the prompt input directly.
        display: bool
            Whereas printing the model answer or not. Default is 'True'.

        Returns
        -------
        output: str
            The model's response to the input.

        Example
        -------
        >>> prompt = "Synthesize this conversation"
        >>> context = f'{conversation}'
        # The model input will be formatted as:
        >>> model_input = context + prompt
        """

        messages = self._preprocess(prompts=prompts, contexts=contexts)
        response = self.pipe(messages, 
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            top_k=50,
                            top_p=0.9,
                            temperature=0.7,
                            repetition_penalty=1.1)

        outputs = self._postprocess(results=response, messages=messages)

        if display: print(*outputs, sep="\n")
        
        if isinstance(prompts, str):
            # If input is a single prompt, returns directly the answer
            return outputs[0]
        # Otherwise, returns a list of answers in the same order as the inputs
        return outputs
    
    def _preprocess(self, prompts: list[str], contexts: list[str], **kwargs):
        """
        Preprocesses the pipeline inputs.
        """

        if isinstance(prompts, str): prompts = [prompts]
        if not isinstance(prompts, list): 
            print(f"LLMEmbeddingSentenceTransformer >> Error: Model's prompt input should be of type 'list[str]', got '{type(prompts)}' instead.")

        if isinstance(contexts, str): 
            contexts = [contexts] * len(prompts)
        if not isinstance(contexts, list): 
            print(f"LLMEmbeddingSentenceTransformer >> Error: Model's context input should be of type 'list[str]', got '{type(contexts)}' instead.")

        messages = []
        for context, prompt in zip(contexts, prompts):
            messages.append(context + '\n' + prompt)

        return messages
    
    def _postprocess(self, results: list[str], messages: list[str]):
        return [result[0]['generated_text'][len(message):] for result, message in zip(results, messages)]

