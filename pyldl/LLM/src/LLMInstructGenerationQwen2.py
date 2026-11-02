import os
import sys
from typing import Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, AutoConfig
from accelerate import init_empty_weights

from .LLM import LLM


class LLMInstructGenerationQwen2(LLM):
    """
    A collection of pretrained models based on Alibaba's Qwen2.5 backbone, fine-tuned for text generation and instruction following.
    Includes built-in support for dynamic LoRA fine-tuning and inference.
    """
    def __init__(self, weights_path: Optional[str] = None, version: str = "0.5B") -> None:
        """
        Initializes a pretrained model based on the Qwen2.5 backbone.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.
        version: str
            The size of the model to load.

        Versions
        --------
        - ``'0.5B'`` _(default)_ : The smallest 0.5 billion parameters version of Qwen 2.5. 
            - Initializes the model with ``'Qwen/Qwen2.5-0.5B-Instruct'`` weights.
            - Highly efficient, suitable for edge devices and rapid prototyping.
        - ``'1.5B'``: The 1.5 billion parameters version.
            - Initializes the model with ``'Qwen/Qwen2.5-1.5B-Instruct'`` weights.
        - ``'3B'``: The 3 billion parameters version.
            - Initializes the model with ``'Qwen/Qwen2.5-3B-Instruct'`` weights.
        - ``'7B'``: The 7 billion parameters version.
            - Initializes the model with ``'Qwen/Qwen2.5-7B-Instruct'`` weights.
        """

        self.version = version
        model_map = {
            "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
            "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
            "3B": "Qwen/Qwen2.5-3B-Instruct",
            "7B": "Qwen/Qwen2.5-7B-Instruct"
        }

        if version in model_map:
            model_name = model_map[version]
        else:
            print(f"LLMInstructGenerationQwen2 >> Warning: Invalid model version '{version}', model '0.5B' will be used instead.")
            self.version = "0.5B"
            model_name = model_map["0.5B"]

        super().__init__(model_name=model_name, weights_path=weights_path)


    def load_model(self, quantization: Optional[str] = None, lora: bool = False, display: bool = False):
        """
        Loads the selected Qwen model, evaluates the device map, applies LoRA adapters if requested, 
        and sets up the text generation pipeline.

        Parameters
        ----------
        quantization: str, None
            Quantizes the model to reduce memory usage. Accepts '4b' or '8b'. 
        lora: bool
            If True, looks for an 'adapter_config.json' in the model folder and loads the weights.
        display: bool
            If True, prints the computed device mapping before loading.
        """
        if quantization in ["8b", "4b"] and torch.cuda.is_available():
            load_in_8bit = (quantization == "8b")
            load_in_4bit = (quantization == "4b")
            dtype_correction = 2.0 if load_in_8bit else 4.0
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit, 
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            low_cpu_mem_usage = True
        else:
            low_cpu_mem_usage = None
            quantization_config = None
            dtype_correction = 2.0

        # Pre-compute the device map using the parent class logic
        config = AutoConfig.from_pretrained(self.model_folder, quantization_config=quantization_config)
        with init_empty_weights(include_buffers=True):
            empty_model = AutoModelForCausalLM.from_config(config)
            self._device_map(model=empty_model, dtype_correction=dtype_correction, display=display)
            del empty_model

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_folder, 
            trust_remote_code=True, 
            quantization_config=quantization_config, 
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=self.device_map,
            dtype=torch.bfloat16
        )

        # LoRA Adapter Loading
        if lora and os.path.exists(os.path.join(self.model_folder, "adapter_config.json")):
            
            # Check if transformers already auto-loaded the adapter natively during from_pretrained
            if hasattr(self.model, "peft_config"):
                print("LLMInstructGenerationQwen2 >> LoRA adapter automatically loaded by transformers.")
                
            else:
                from peft import PeftModel
                print("LLMInstructGenerationQwen2 >> LoRA adapter detected. Wrapping model with PeftModel...")
                
                # We MUST use PeftModel here instead of load_adapter() to access merge_and_unload()
                self.model = PeftModel.from_pretrained(self.model, self.model_folder)
            
            # # Merging bakes LoRA weights into the base model for faster inference.
            # if not quantization:
            #     print("LLMInstructGenerationQwen2 >> Merging LoRA weights into base model...")
            #     # This will now work, as self.model is officially a PeftModel object
            #     self.model = self.model.merge_and_unload()
            # else:
            #     print("LLMInstructGenerationQwen2 >> Note: Cannot merge LoRA weights into a quantized model. Proceeding unmerged.")

                
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_folder, 
            clean_up_tokenization_spaces=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            dtype=torch.bfloat16,
            device_map=self.device_map
        )
                
        return None

    
    def evaluate_model(self, prompts: list[str] | str, contexts: list[str] | str = "", max_tokens: int = 1000, display: bool = False, **kwargs):
        """
        Evaluates a prompt and returns the model answer using ChatML formats natively.

        Parameters
        ----------
        prompts: str or list of str
            The user inputs to generate responses for.
        contexts: str or list of str
            System instructions to prepend to the conversation.
        max_tokens: int
            Maximum number of new tokens to generate.
        display: bool
            If True, prints the generated outputs to the console.
        """
        messages_list = self._preprocess(prompts=prompts, contexts=contexts)
        
        # Pipeline natively accepts dictionaries and handles the chat template mapping.
        # return_full_text=False prevents the prompt from being echoed in the final output.
        responses = self.pipe(
            messages_list, 
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
            return_full_text=False 
        )

        outputs = self._postprocess(results=responses)

        if display: 
            print(*outputs, sep="\n\n")
        
        if isinstance(prompts, str):
            return outputs[0]
        return outputs

    
    def _preprocess(self, prompts: list[str] | str, contexts: list[str] | str, **kwargs):
        """
        Preprocesses inputs into a structured list of conversational dictionaries for the pipeline.
        """
        if isinstance(prompts, str): 
            prompts = [prompts]
        if isinstance(contexts, str): 
            contexts = [contexts] * len(prompts)

        messages_list = []
        for context, prompt in zip(contexts, prompts):
            chat = []
            if context:
                chat.append({"role": "system", "content": context})
            chat.append({"role": "user", "content": prompt})
            messages_list.append(chat)

        return messages_list

    
    def _postprocess(self, results: list[list[dict]]):
        """
        Extracts and cleans the generated text from the pipeline output format.
        """
        return [res[0]['generated_text'].strip() for res in results]


    def finetune(
        self, 
        data: list[dict[Literal["system", "user", "assistant"], str]], 
        output_dir: Optional[str] = None, 
        epochs: int = 3, 
        batch_size: int = 4, 
        learning_rate: float = 2e-4
    ):
        """
        Fine-tunes the loaded model dynamically using LoRA via the modern SFTConfig.
        
        Parameters
        ----------
        data: list of dicts
            Must follow the conversational structure: 
            [{"system": "...", "user": "...", "assistant": "..."}]
        output_dir: str, None
            Path to save the LoRA adapter weights. Defaults to the base model folder.
        epochs: int
            Number of training epochs.
        batch_size: int
            Per-device training batch size.
        learning_rate: float
            Optimizer learning rate.
        """
        if output_dir is None:
            output_dir = self.model_folder

        try:
            from datasets import Dataset
            from peft import LoraConfig, get_peft_model
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            raise ImportError("Training requires peft, trl, and datasets. Run: `pip install peft trl datasets`")

        print("LLMInstructGenerationQwen2 >> Formatting dataset for modern TRL processing...")
        
        # Modern `trl` natively processes a "messages" column, eliminating the need for manual string templates.
        formatted_data = []
        for row in data:
            messages = []
            if row.get("system"): messages.append({"role": "system", "content": row["system"]})
            if row.get("user"): messages.append({"role": "user", "content": row["user"]})
            if row.get("assistant"): messages.append({"role": "assistant", "content": row["assistant"]})
            formatted_data.append({"messages": messages})
            
        dataset = Dataset.from_list(formatted_data)

        print("LLMInstructGenerationQwen2 >> Injecting LoRA adapters...")
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Modern SFTConfig replaces standard TrainingArguments and handles sequence lengths natively
        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            logging_steps=5,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            save_strategy="epoch",
            max_length=1024,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=training_args,
        )

        print("LLMInstructGenerationQwen2 >> Beginning training loop...")
        trainer.train()
        
        # Save only the adapter weights (a few megabytes) instead of the whole model
        trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"LLMInstructGenerationQwen2 >> Training complete! Adapters saved to {output_dir}")

        # Sync pipeline back up to the trained model
        self.pipe.model = self.model