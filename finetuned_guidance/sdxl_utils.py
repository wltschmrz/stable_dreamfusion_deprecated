import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,   # 혹은 EulerDiscreteScheduler 등
)
from safetensors.torch import load_file
from torchvision.utils import save_image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.utils import deprecate

class StableDiffusionXL(nn.Module):
    def __init__(
            self, device="cuda", fp16=False, vram_O=False,
            hf_key="stabilityai/stable-diffusion-xl-base-1.0",
            lora_dir = "/workspace/stable_dreamfusion_deprecated/finetuned_guidance/checkpoint-1000",
            t_range=[0.02, 0.98],
            scheduler_type="DDIM",
            **kwargs
            ):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32
        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            base_model_id = hf_key; print(f'[INFO] using hugging face custom model key: {hf_key}')
        else:
            raise ValueError(f'Stable-diffusion {hf_key} not supported.')

        pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=self.precision_t, **kwargs)

        # 2) LoRA 가중치 로드
        if lora_dir is not None:
            print(f"[INFO] Loading LoRA from {lora_dir}")
            pipe.load_lora_weights(lora_dir, adapter_name="subject")

        # 3) Textual Inversion(learned_embeds.safetensors) 로드
        ti_dir = lora_dir
        if ti_dir is not None:
            print(f"[INFO] Loading textual inversion from {ti_dir}")
            inserting_tokens = ["<man>"]
            state_dict = load_file(f"{ti_dir}/learned_embeds.safetensors")
            pipe.load_textual_inversion(state_dict["clip_l"], token=inserting_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            pipe.load_textual_inversion(state_dict["clip_g"], token=inserting_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

        # 4) Scheduler 교체 (DDIM or EulerDiscrete 등)
        if scheduler_type == "DDIM":
            self.scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler", torch_dtype=self.precision_t)
        else:
            raise ValueError  # 기타 PNDMScheduler, DPMSolverScheduler 등

        if vram_O:  ## 오류나면 그냥 지워
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = getattr(pipe, "tokenizer_2", None)
        self.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        self.text_encoder_2
        self.unet = pipe.unet

        self.do_classifier_free_guidance = getattr(pipe, "do_classifier_free_guidance", True)
        # self.guidance_rescale = getattr(pipe, "guidance_rescale", 0.0)
        self.encode_prompt = pipe.encode_prompt
        # self._get_add_time_ids = pipe._get_add_time_ids
        self.default_sample_size = getattr(pipe, "default_sample_size", 128)
        self.vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
        height = width = self.default_sample_size * self.vae_scale_factor
        self.original_size = (height, width)
        self.target_size = (height, width)
        self.cross_attention_kwargs = 

        # alphas_cumprod 등 DreamFusion용 변수
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        try:
            self.alphas = self.scheduler.alphas_cumprod.to(device)
        except:
            # Euler 등은 alphas_cumprod가 없을 수 있음
            self.alphas = None; print("[WARN] This scheduler does not have `alphas_cumprod`. Train step must be adapted!")

        del pipe; print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2 = None,

        num_inference_steps = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
        ):

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
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






    def train_step(
            self,
            text_embeddings,
            pred_rgb,
            guidance_scale=100,
            as_latent=False,
            grad_scale=1,
            save_guidance_path=None
            ):
        print(text_embeddings.shape)  # 2,77,2048 (uncond, cond)

        if as_latent:
            latents = F.interpolate(pred_rgb, (128, 128), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (1024, 1024), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}  # [2,1280] [2,6]
            noise_pred = self.unet(
                latent_model_input,
                tt,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        '''
        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # TODO: also denoise all-the-way

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)
        '''
        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss





if __name__ == "__main__":
    pipe = StableDiffusionXL(
        device="cuda", fp16=False, vram_O=False,
        )
