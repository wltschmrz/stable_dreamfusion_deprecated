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
        self.guidance_rescale = getattr(pipe, "guidance_rescale", 0.0)

        self.encode_prompt = pipe.encode_prompt
        self._get_add_time_ids = pipe._get_add_time_ids

        # print(self.vae.config.scaling_factor)  # 0.13025

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
        # print(original_size, target_size)  ####
        if not original_size==target_size==(512,512):
            original_size = target_size = (512,512)
        self.original_size = self.target_size = (512,512)

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
        
        returndict = {
            "prompt_embeds": prompt_embeds,                 # [2, 77, 2048]
            "add_text_embeds": add_text_embeds,             # [2, 1280]
            "add_time_ids": add_time_ids,                   # [2, 6]
            "lora_prompt_embeds": lora_prompt_embeds,       # [1, 77, 2048]
            "lora_add_text_embeds": lora_add_text_embeds,   # [1, 1280]
            "lora_add_time_ids": lora_add_time_ids,         # [2, 6]
        }

        # for k in returndict:
        #     print(f"{k}: {returndict[k].shape}")

        # 6) 반환 (unet forward에 필요한 정보)
        return returndict

    def encode_imgs(self, imgs):
        """
        imgs: [B,3,H,W] in [0,1] range
        """
        dtype = next(self.vae.parameters()).dtype
        imgs = imgs.to(device=self.device, dtype=dtype)
        imgs = 2*imgs -1
        posterior = self.vae.encode(imgs).latent_dist
        # SDXL vae.config.scaling_factor는 0.13025임
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

        # for key in text_embeddings:
        #     print(f'{key}: {text_embeddings[key].shape}')

        self.guidance_scale = guidance_scale
        self.guidance_scale_lora = guidance_scale_lora
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64,64), mode='bilinear', align_corners=False)*2 -1
            # print(latents.shape)
            # print("as latent")
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)  ####
            latents = self.encode_imgs(pred_rgb_512)
            # print(pred_rgb_512.shape)
            # print(latents.shape)
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
            # print(latent_model_input.shape)

            prompt_embeds = text_embeddings["prompt_embeds"]  # [4, 77, 2048]
            add_text_embeds = text_embeddings["add_text_embeds"]  # [4, 1280]
            add_time_ids = text_embeddings["add_time_ids"]  # [4, 6]
            lora_prompt_embeds = text_embeddings["lora_prompt_embeds"]  # [2, 77, 2048]
            lora_add_text_embeds = text_embeddings["lora_add_text_embeds"]  # [2, 1280]
            lora_add_time_ids = text_embeddings["lora_add_time_ids"]  # [4, 6]

            _, prompt_embeds = torch.split(prompt_embeds, 2, dim=0)
            _, add_text_embeds = torch.split(add_text_embeds, 2, dim=0)
            _, add_time_ids = torch.split(add_time_ids, 2, dim=0)
            _, lora_prompt_embeds = torch.split(lora_prompt_embeds, 1, dim=0)
            _, lora_add_text_embeds = torch.split(lora_add_text_embeds, 1, dim=0)
            _, _, _, lora_add_time_ids = torch.split(lora_add_time_ids, 1, dim=0)

            if self.original_size == (512, 512):
                lora_latent_model_input = latent_model_input[0].view(1, 4, 64, 64)
            elif self.original_size == (1024, 1024):
                lora_latent_model_input = latent_model_input[0].view(1, 4, 128, 128)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            lora_added_cond_kwargs = {"text_embeds": lora_add_text_embeds, "time_ids": lora_add_time_ids}

            # tt = torch.cat([t]*2)
            cross_attention_kwargs_pre = {"scale": 0.0}

            # print(f"[DEBUG] latent_model_input shape: {latent_model_input.shape}")
            # print(f"[DEBUG] prompt_embeds shape: {prompt_embeds.shape}")
            # print(f"[DEBUG] t shape: {t.shape}")
            # print(f"[DEBUG] latent_model_input shape: {lora_latent_model_input.shape}")
            # print(f"[DEBUG] prompt_embeds shape: {lora_prompt_embeds.shape}")
            # print(f"[DEBUG] t shape: {t.shape}")

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



    @torch.no_grad()
    def my_gen(
        self,
        prompt: Union[str, List[str]] = None,                                       # base prompt
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_ti: Union[str, List[str]] = None,                                    # prompt
        prompt_2_ti: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,                                                # 7.5
        guidance_scale_lora: float = 5.0,                                           # rg scale=3.0
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # None
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image=None,  # : Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,                    # {"scale": 1.0}
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

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_scale_lora = guidance_scale_lora
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._denoising_end = denoising_end

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
            self._cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

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
        
        

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_text_embeds_ti = pooled_prompt_embeds_ti
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        lora_prompt_embeds = prompt_embeds_ti
        lora_add_text_embeds = add_text_embeds_ti
        lora_add_time_ids = add_time_ids
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        lora_prompt_embeds = lora_prompt_embeds.to(device)
        lora_add_text_embeds = lora_add_text_embeds.to(device)
        lora_add_time_ids = lora_add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        
        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                lora_latent_model_input = latent_model_input[0].view(1, 4, 128, 128)
                # print(latent_model_input.size())
                # lora_latent_model_input = latents
                # lora_latent_model_input = self.scheduler.scale_model_input(lora_latent_model_input, t)
                # print(lora_latent_model_input.size())
                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                lora_added_cond_kwargs = {"text_embeds": lora_add_text_embeds, "time_ids": lora_add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                

                print(f"[DEBUG] latent_model_input shape: {latent_model_input.shape}")
                print(f"[DEBUG] prompt_embeds shape: {prompt_embeds.shape}")
                print(f"[DEBUG] t shape: {t.shape}")
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
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=lora_added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = self.guidance_scale * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond
                    noise_pred += guidance_scale_lora * (noise_pred_lora_text - noise_pred_text)
                    # noise_pred_lora_text + 

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)












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
