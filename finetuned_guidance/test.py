import torch
import os
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import random

output_path = ".png"
lora_dir = "/workspace/stable_dreamfusion_deprecated/finetuned_guidance/checkpoint-1000"
assert os.path.exists(lora_dir), f"Error: {lora_dir} does not exist."

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")    
pipe.load_lora_weights(lora_dir, adapter_name='subject')

inserting_tokens = ["<man>"] 
state_dict = load_file(lora_dir+"/learned_embeds.safetensors")
pipe.load_textual_inversion(state_dict["clip_l"], token=inserting_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
pipe.load_textual_inversion(state_dict["clip_g"], token=inserting_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

prompts = [
  "A color photo of <man>'s head with hair, random solid background, sharp details, high contrast, no haze effect, correct gamma levels, side view",
  "A color photo of <man>'s head with hair, random solid background, sharp details, high contrast, no haze effect, correct gamma levels, front view",
  "A color photo of <man>'s head with hair, random solid background, sharp details, high contrast, no haze effect, correct gamma levels, back view",
  "A color photo of <man>'s head with hair, random solid background sharp details, high contrast, no haze effect, correct gamma levels",
]

for ii in range(10):
    seed = random.randint(1, 1000)
    generator = torch.Generator("cuda").manual_seed(seed)
    for i, prompt in enumerate(prompts):
        image = pipe(
                    prompt=prompt,
                    generator=generator,
                    cross_attention_kwargs={"scale": 1.0},
                    guidance_scale=7.5,
                ).images[0]

        if 'side ' in prompt:
            output_path_ = f"{seed}_side"+output_path
        elif 'back ' in prompt:
            output_path_ = f"{seed}_back"+output_path
        elif 'front ' in prompt:
            output_path_ = f"{seed}_front"+output_path
        else:
            output_path_ = f"{seed}_plain"+output_path

        output_path_ = f"./finetuned_guidance/results/"+output_path_
        image.save(output_path_)
        print(f"Image saved at {output_path_}")


# model_key = "stabilityai/stable-diffusion-2-1-base"
# precision_t = torch.float32
# pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t).to("cuda")    
# for i, prompt in enumerate(prompts):
#     image = pipe(
#                 prompt=prompt,
#                 generator=generator,
#                 # cross_attention_kwargs={"scale": 1.0},
#                 guidance_scale=7.5,
#             ).images[0]
#     output_path_ = f"./finetuned_guidance/results/normal{i}.png"
#     image.save(output_path_)
