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

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps = None,
    device = None,
    timesteps = None,
    **kwargs,
):
    """
    스케줄러의 `set_timesteps`를 호출한 후 timesteps를 가져오며, 커스텀 timesteps도 지원한다.

    반환값 (Returns)
    - `Tuple[torch.Tensor, int]`: timesteps 스케줄과 Inference 단계 수.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class FinetunedSDXL(nn.Module):
    def __init__(
            self,
            device,
            fp16=True,
            vram_O=False,
            # SDXL + DCO + LoRA + TI 관련 파라미터
            base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
            lora_dir=None,
            textual_inversion_dir=None,
            t_range=[0.02, 0.98],
            # 원하는 scheduler를 지정. DDIM, Euler 등
            scheduler_type="DDIM",  # "Euler", "PNDM", ...
            # 기타 DreamFusion 옵션
            **kwargs
        ):
        super().__init__()
        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32
        print(f"[INFO] Loading Finetuned SDXL with base={base_model_id}")

        # 1) 먼저 SDXL 파이프라인 로드 (RGPipe 쓸 수도 있음)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_id,
            torch_dtype=self.precision_t,
            **kwargs
        )
        # vram 옵션
        if vram_O:
            ## 필요시 offload. 예: pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            ## 오류나면 그냥 지워
        else:
            pipe.to(device)

        # 2) LoRA 가중치 로드
        if lora_dir is not None:
            print(f"[INFO] Loading LoRA from {lora_dir}")
            pipe.load_lora_weights(lora_dir, adapter_name="subject")

        # 3) Textual Inversion(learned_embeds.safetensors) 로드
        if textual_inversion_dir is not None:
            print(f"[INFO] Loading textual inversion from {textual_inversion_dir}")
            inserting_tokens = ["<man>"]
            state_dict = load_file(f"{textual_inversion_dir}/learned_embeds.safetensors")
            pipe.load_textual_inversion(
                state_dict["clip_l"],
                token=inserting_tokens,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer
            )
            pipe.load_textual_inversion(
                state_dict["clip_g"],
                token=inserting_tokens,
                text_encoder=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer_2
            )

        # 4) Scheduler 교체 (DDIM or EulerDiscrete 등)
        if scheduler_type == "DDIM":
            self.scheduler = DDIMScheduler.from_pretrained(
                base_model_id,
                subfolder="scheduler",
                torch_dtype=self.precision_t
            )
        elif scheduler_type == "Euler":
            from diffusers import EulerDiscreteScheduler
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                base_model_id,
                subfolder="scheduler",
                torch_dtype=self.precision_t
            )
        else:
            # 기타 PNDMScheduler, DPMSolverScheduler 등
            pass
            
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = getattr(pipe, "tokenizer_2", None)
        self.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        self.text_encoder_2
        self.unet = pipe.unet
        self.scheduler

        self.do_classifier_free_guidance = getattr(pipe, "do_classifier_free_guidance", True)
        self.default_sample_size = getattr(pipe, "default_sample_size", 128)
        self.vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
        
        self.encode_prompt = pipe.encode_prompt
        self._get_add_time_ids = pipe._get_add_time_ids

        # alphas_cumprod 등 DreamFusion용 변수
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        try:
            self.alphas = self.scheduler.alphas_cumprod.to(device)
        except:
            # Euler 등은 alphas_cumprod가 없을 수 있음
            self.alphas = None
            print("[WARN] This scheduler does not have `alphas_cumprod`. Train step must be adapted!")

        del pipe
        print("[INFO] Done initializing FinetunedSDXL.")

    @torch.no_grad()
    def get_text_embeds(
        self,
        prompt: str = None,                     # base prompt
        prompt_ti: str = None,                  # textual inversion prompt
        prompt_2: str = None,
        prompt_2_ti: str = None,
        negative_prompt: str = None,
        negative_prompt_2: str = None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 5.0,            # CFG 스케일
        guidance_scale_lora: float = 5.0,       # LoRA용 스케일
        cross_attention_kwargs: dict = None,    # {"scale": 1.0} 등
        clip_skip: int = None,
        height: int = None,
        width: int = None,
        guidance_rescale: float = 0.0,
        generator=None,
        original_size=None,
        target_size=None,
        **kwargs,
    ):
        device = self.device
        do_classifier_free_guidance = self.do_classifier_free_guidance
        cross_attention_kwargs = cross_attention_kwargs or {}

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        print(original_size, target_size)  ####
        if not original_size==target_size==(512,512):
            original_size = target_size = (512,512)

        # 1) prompt 개수 확인
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            prompt_ti = [prompt_ti]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        # 2) LoRA 스케일 (my_gen에서 cross_attention_kwargs 사용)
        self._guidance_scale = guidance_scale
        self._guidance_scale_lora = guidance_scale_lora
        self._cross_attention_kwargs = cross_attention_kwargs
        self._clip_skip = clip_skip

        # 3) 텍스트 인코딩( base prompt vs. TI prompt )  
        #   - prompt : base prompt
        #   - prompt_ti : textual inversion prompt
        lora_scale = cross_attention_kwargs.get("scale", None)

        (
            prompt_embeds_ti,
            negative_prompt_embeds_ti,
            pooled_prompt_embeds_ti,
            negative_pooled_prompt_embeds_ti,
        ) = self.encode_prompt(
            prompt=prompt_ti,
            prompt_2=prompt_2_ti,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=None,
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
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=None,
        )

        # 4) additional embeddings (micro-conditioning용) 설정
        add_text_embeds = pooled_prompt_embeds
        add_text_embeds_ti = pooled_prompt_embeds_ti
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            (0,0),
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        # if negative_original_size is not None and negative_target_size is not None:
        #     negative_add_time_ids = self._get_add_time_ids(
        #         negative_original_size,
        #         negative_crops_coords_top_left,
        #         negative_target_size,
        #         dtype=prompt_embeds.dtype,
        #         text_encoder_projection_dim=text_encoder_projection_dim,
        #     )
        # else:
        negative_add_time_ids = add_time_ids  #

        # 5) LoRA prompt 임베딩 / classifier-free guidance 구성
        if self.do_classifier_free_guidance:
            # (uncond, cond) concat
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        lora_prompt_embeds = prompt_embeds_ti.to(device)
        lora_add_text_embeds = add_text_embeds_ti.to(device)
        lora_add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        
        # 6) 반환 (unet forward에 필요한 정보)
        return {
            "prompt_embeds": prompt_embeds,
            "add_text_embeds": add_text_embeds,
            "add_time_ids": add_time_ids,
            "lora_prompt_embeds": lora_prompt_embeds,
            "lora_add_text_embeds": lora_add_text_embeds,
            "lora_add_time_ids": lora_add_time_ids,
        }

    def encode_imgs(self, imgs):
        """
        imgs: [B,3,H,W] in [0,1] range
        """
        dtype = next(self.vae.parameters()).dtype
        imgs = imgs.to(device=self.device, dtype=dtype)
        imgs = 2*imgs -1
        posterior = self.vae.encode(imgs).latent_dist
        # SDXL vae.config.scaling_factor가 0.13025일 수도 있음
        print(self.vae.config.scaling_factor)  ####
        scale = getattr(self.vae.config, "scaling_factor", 0.13025)
        latents = posterior.sample() * scale
        return latents

    def decode_latents(self, latents):
        scale = getattr(self.vae.config, "scaling_factor", 0.13025)
        latents = latents / scale
        imgs = self.vae.decode(latents).sample
        imgs = (imgs*0.5+0.5).clamp(0,1)
        return imgs

    def train_step(
        self, 
        text_embeddings, # dict: [B, seq_len, hidden_dim]
        pred_rgb,        # [B, 3, H, W]
        guidance_scale=100,
        guidance_scale_lora=40,
        as_latent=False,
        grad_scale=1,
        save_guidance_path:Path=None
    ):
        self.guidance_scale = guidance_scale
        self.guidance_scale_lora = guidance_scale_lora
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64,64), mode='bilinear', align_corners=False)*2 -1
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (1024, 1024), mode='bilinear', align_corners=False)  ####
            latents = self.encode_imgs(pred_rgb_512)

        # t in [min_step, max_step]
        t = torch.randint(self.min_step, self.max_step+1, (latents.shape[0],), device=self.device, dtype=torch.long)

        with torch.no_grad():
            # 9. Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                batch_size = 1
                num_images_per_prompt = 1
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=self.device, dtype=latents.dtype)

            # add noise
            if not hasattr(self.scheduler, "add_noise"):
                raise ValueError(f"Scheduler {self.scheduler.__class__.__name__} has no add_noise method!")
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # unet forward with CFG
            latent_model_input = torch.cat([latents_noisy]*2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            prompt_embeds = text_embeddings["prompt_embeds"]
            lora_prompt_embeds = text_embeddings["lora_prompt_embeds"]
            add_text_embeds = text_embeddings["add_text_embeds"]
            add_time_ids = text_embeddings["add_time_ids"]
            lora_add_text_embeds = text_embeddings["lora_add_text_embeds"]
            lora_add_time_ids = text_embeddings["lora_add_time_ids"]

            lora_latent_model_input = latent_model_input[0].view(1, 4, 128, 128)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            lora_added_cond_kwargs = {"text_embeds": lora_add_text_embeds, "time_ids": lora_add_time_ids}

            # tt = torch.cat([t]*2)
            cross_attention_kwargs_pre = {"scale": 0.0}

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs_pre,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred_lora_text = self.unet(
                lora_latent_model_input,
                t,
                encoder_hidden_states=lora_prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self._cross_attention_kwargs,
                added_cond_kwargs=lora_added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = self.guidance_scale * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond
                noise_pred += self.guidance_scale_lora * (noise_pred_lora_text - noise_pred_text)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

        if self.alphas is None:
            # EulerDiscreteScheduler 등에서 alphas_cumprod가 없는 경우
            raise ValueError("No alphas_cumprod found! Must adapt train_step for EulerDiscreteScheduler or switch to DDIM.")
        w = (1 - self.alphas[t])  # [B]
        grad = grad_scale * w[:,None,None,None]*(noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]
        return loss

    def train_step_perpneg(
        self,
        text_embeddings,
        weights,
        pred_rgb,
        guidance_scale=100,
        as_latent=False,
        grad_scale=1,
        save_guidance_path:Path=None
    ):
        """
        perpneg_utils.py 로직과 유사. EulerDiscreteScheduler 사용 시 동일한 문제가 생길 수 있음.
        """
        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1

        if as_latent:
            latents = F.interpolate(pred_rgb, (64,64))*2 -1
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512,512))
            latents = self.encode_imgs(pred_rgb_512)

        t = torch.randint(self.min_step, self.max_step+1, (B,), device=self.device, dtype=torch.long)
        with torch.no_grad():
            if not hasattr(self.scheduler, "add_noise"):
                raise ValueError("Scheduler has no add_noise method! Switch to e.g. DDIM.")
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            latent_model_input = torch.cat([latents_noisy]*(1+K))
            tt = torch.cat([t]*(1+K))
            unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K,1,1,1)

            # perpneg aggregator
            noise_pred = noise_pred_uncond + guidance_scale*weighted_perpendicular_aggregator(delta_noise_preds, weights, B)

        if self.alphas is None:
            raise ValueError("No alphas_cumprod found for this scheduler!")
        w = (1 - self.alphas[t])
        grad = grad_scale*w[:,None,None,None]*(noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            pass

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]
        return loss
