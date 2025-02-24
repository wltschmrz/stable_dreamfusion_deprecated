import torch
import os
from safetensors.torch import load_file
from reward_guidance import RGPipe

lora_dir = "/workspace/stable_dreamfusion_depre/finetuned_guidance/checkpoint-1000"

pipe = RGPipe.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights(lora_dir, adapter_name='subject')

inserting_tokens = ["<man>"]  
state_dict = load_file(lora_dir+"/learned_embeds.safetensors")
pipe.load_textual_inversion(state_dict["clip_l"], token=inserting_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
pipe.load_textual_inversion(state_dict["clip_g"], token=inserting_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

generator = torch.Generator("cuda").manual_seed(seed)

prompt_= '<man>'
base_prompt = 'man'

rg_scale = 3.0
image = pipe.my_gen(
    prompt=base_prompt,
    prompt_ti=prompt_, 
    generator=generator,
    cross_attention_kwargs={"scale": 1.0},
    guidance_scale=7.5,
    guidance_scale_lora=rg_scale,
    ).images[0]

prompt=base_prompt
prompt_ti=prompt_
prompt_2 = None
prompt_2_ti = None
cross_attention_kwargs={"scale": 1.0}
guidance_scale=7.5
guidance_scale_lora=rg_scale
num_images_per_prompt = 1


device = pipe._execution_device

lora_scale = (
    cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
)

(
    prompt_embeds_ti,
    negative_prompt_embeds_ti,
    pooled_prompt_embeds_ti,
    negative_pooled_prompt_embeds_ti,
) = pipe.encode_prompt(
    prompt=prompt_ti,
    prompt_2=prompt_2_ti,
    device=device,
    num_images_per_prompt=num_images_per_prompt,
    do_classifier_free_guidance=self.do_classifier_free_guidance,
    negative_prompt=negative_prompt,
    negative_prompt_2=negative_prompt_2,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    lora_scale=lora_scale,
    clip_skip=self.clip_skip,
)

(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
) = self.encode_prompt(
    prompt=prompt,
    prompt_2=prompt_2,
    device=device,
    num_images_per_prompt=num_images_per_prompt,
    do_classifier_free_guidance=self.do_classifier_free_guidance,
    negative_prompt=negative_prompt,
    negative_prompt_2=negative_prompt_2,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    lora_scale=lora_scale,
    clip_skip=self.clip_skip,
)