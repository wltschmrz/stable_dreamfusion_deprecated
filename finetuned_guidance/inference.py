import torch
import os
from safetensors.torch import load_file
from reward_guidance import RGPipe
import random

output_path = ".png"
lora_dir = "/workspace/stable_dreamfusion_deprecated/finetuned_guidance/checkpoint-1000"
assert os.path.exists(lora_dir), f"Error: {lora_dir} does not exist."

pipe = RGPipe.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")    
pipe.load_lora_weights(lora_dir, adapter_name='subject')

inserting_tokens = ["<man>"] 
state_dict = load_file(lora_dir+"/learned_embeds.safetensors")
pipe.load_textual_inversion(state_dict["clip_l"], token=inserting_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
pipe.load_textual_inversion(state_dict["clip_g"], token=inserting_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

prompts = [
  "A DSLR photo of <man>'s head with hair, side view", # (in drowing style) prompt including new tokens
  "A DSLR photo of <man>'s head with hair, front view",
  "A DSLR photo of <man>'s head with hair, back view",
  "A DSLR photo of <man>'s head with hair",
]

base_prompts = [
  "A DSLR photo of man's head with hair, side view", # prompt without new tokens
  "A DSLR photo of man's head with hair, front view",
  "A DSLR photo of man's head with hair, back view",
  "A DSLR photo of man's head with hair",
]

seed = random.randint(1, 1000)
generator = torch.Generator("cuda").manual_seed(seed)

rg_scale = 0.0 # rg scale. 0.0 for original CFG sampling
for i, (prompt, base_prompt) in enumerate(zip(prompts, base_prompts)):
  if rg_scale > 0.0:
    image = pipe.my_gen(
        prompt=base_prompt,
        prompt_ti=prompt, 
        generator=generator,
        cross_attention_kwargs={"scale": 1.0},
        guidance_scale=7.5,
        guidance_scale_lora=rg_scale,
        ).images[0]
  else:
    image = pipe(
        prompt=prompt,
        generator=generator,
        cross_attention_kwargs={"scale": 1.0},
        guidance_scale=7.5,
        ).images[0]

  if 'side' in prompt:
    output_path_ = f"side_{seed}"+output_path
  elif 'back' in prompt:
    output_path_ = f"back_{seed}"+output_path
  elif 'front' in prompt:
    output_path_ = f"front_{seed}"+output_path
  else:
    output_path_ = f"plain_{seed}"+output_path

  output_path_ = f"./finetuned_guidance/results/"+output_path_
  image.save(output_path_)
  print(f"Image saved at {output_path_}")



