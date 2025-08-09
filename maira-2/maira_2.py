from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
from PIL import Image
from huggingface_hub import login
import torch

hf_token = "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"
login(hf_token)

model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval()
model = model.to(device)

image = Image.open("7d5c70800f518f6584e25b2a88238a22.png").convert("RGB")

processed_inputs = processor.format_and_preprocess_reporting_input(
    current_frontal=image,
    current_lateral=None,
    prior_frontal=None,  
    indication=None,
    technique=None,
    comparison=None,
    prior_report=None,  
    return_tensors="pt",
    get_grounding=True,  
)

processed_inputs = processed_inputs.to(device)
with torch.no_grad():
    output_decoding = model.generate(
        **processed_inputs,
        max_new_tokens=450, 
        use_cache=True,
    )
prompt_length = processed_inputs["input_ids"].shape[-1]
decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
print("Parsed prediction:", prediction)
