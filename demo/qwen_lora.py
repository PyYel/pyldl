from typing import Literal
from pathlib import Path

from pyldl.LLM import LLMInstructGenerationQwen2

model = LLMInstructGenerationQwen2(weights_path=Path(__file__).parent)

model.load_model(lora=True)
# response = model.evaluate_model(prompts="who are you?")
# print(response)


data: list[dict[Literal['system', 'user', 'assistant'], str]] = [
    {"user": "What is gravity?", "assistant": "Arrr, gravity be the anchor that keeps yer boots on the deck!"},
    {"user": "Hello!", "assistant": "Ahoy there, matey!"}
]
model.finetune(data=data, learning_rate=0.001, epochs=2)
response = model.evaluate_model(prompts="who are you?", max_tokens=100)
print(response)

