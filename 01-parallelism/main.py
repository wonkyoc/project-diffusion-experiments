#!/home/wonkyoc/miniconda3/bin/python
import gc
import torch
#import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, \
        PNDMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import make_image_grid
from diffusers.utils import logging

import torch.multiprocessing as mp
from tqdm.auto import tqdm
import time
import os
import argparse


class Inference():
    def __init__(self, args):
        self.debug = True

        # Hyperparameters
        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size
        self.threads = args.threads
        self.prompt = args.prompt   # single prompt
        self.prompts = [self.prompt] * self.batch_size
        self.num_inference_steps = args.steps
        self.height, self.width = 512, 512
        self.guidance_scale = 7.5

        # logging
        self.total_cond_time = 0
        self.total_uncond_time = 0
        self.total_unet_time = 0
        self.total_denoise_time = 0
        self.total_vae_time = 0

        # pipeline
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.scheduler = None

        # torch setting
        torch.set_num_threads(self.threads)

    def load_models(self):
        # Model
        model_id = "runwayml/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", 
                use_safetensors=True)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet",
                use_safetensors=True)
        #scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

        if self.device == "cpu":
            self.unet.share_memory()
            self.vae.share_memory()
            self.text_encoder.share_memory()

    def run(self):
        gc.collect()
       
        if self.device == "mps":
            self.vae.to(self.device)
            self.text_encoder.to(self.device)
            self.unet.to(self.device)
        
        # move here: err: cannot pickle 'torch._C.Generator'
        self.generator = [torch.Generator(self.device).manual_seed(i) for i in range(self.batch_size)]

        # Conditional text prompt
        text_input = self.tokenizer(self.prompts, padding="max_length", 
                max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        self.total_cond_time += time.time() - start

        # Unconditional text prompt
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * self.batch_size, padding="max_length", max_length=max_length, return_tensors="pt")

        start = time.time()
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.total_uncond_time += time.time() - start

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # create random latents
        shape = (1, self.unet.config.in_channels, self.height // 8, self.width // 8)
        latents = [
                torch.randn(shape, generator=self.generator[i], device=self.device)
                for i in range(self.batch_size)
                ]
        latents = torch.cat(latents, dim=0)
        latents = latents * self.scheduler.init_noise_sigma

        # set time step
        self.scheduler.set_timesteps(self.num_inference_steps)

        # denoise
        _total_denoise_time = 0
        for t in tqdm(self.scheduler.timesteps):

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            start = time.time()
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            _total_denoise_time += time.time() - start

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            avg_denoise_time = _total_denoise_time / len(self.scheduler.timesteps)
        self.total_unet_time += _total_denoise_time
        self.total_denoise_time += avg_denoise_time



        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        start = time.time()
        with torch.no_grad():
            decoded_images = self.vae.decode(latents).sample
        self.total_vae_time += time.time() - start

        self.save_image(decoded_images)
        self.save_log()

        ## end ##

    def _make_image(self, i):
        i = (i / 2 + 0.5).clamp(0, 1).squeeze()
        i = (i.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(i)

    def save_image(self, _images):
        self.image_name = f"image-{os.getpid()}"

        images = [self._make_image(i) for i in _images]
        if self.batch_size == 1:
            images = make_image_grid(images, self.batch_size, self.batch_size)
        elif self.batch_size == 3:
            images = make_image_grid(images, 1, 3)
        else:
            rest = int(self.batch_size / 2)
            images = make_image_grid(images, 2, int(self.batch_size / 2))

        images.save(f"{self.image_name}.png")
        gc.collect()

    def save_log(self):
        if self.debug == True:
            self.log_file = f"debug-pid{os.getpid()}-{self.device}-t{self.threads}-b{self.batch_size}-p{self.args.num_processes}.log"
        else:
            self.log_file = f"pid{os.getpid()}-{self.device}-t{self.threads}-b{self.batch_size}-p{self.args.num_processes}.log"

        avg_cond_time = self.total_cond_time / self.args.iteration
        avg_uncond_time = self.total_uncond_time / self.args.iteration
        avg_unet_time = self.total_unet_time / self.args.iteration
        avg_denoise_time = self.total_denoise_time / self.args.iteration
        avg_vae_time = self.total_vae_time / self.args.iteration
        with open(self.log_file, "w") as f:
            f.write(f"config: device={self.device} bs={self.batch_size} threads={self.threads} steps={self.num_inference_steps} procs={self.args.num_processes}\n")
            f.write(f"avg_cond_time (all steps)={avg_cond_time}\n")
            f.write(f"avg_uncond_time (all steps)={avg_uncond_time}\n")
            f.write(f"avg_unet_time (all steps)={avg_unet_time}\n")
            f.write(f"avg_denoise_time (per step)={avg_denoise_time}\n")
            f.write(f"avg_vae_time={avg_vae_time}\n")

def run_process(inst):
    inst.run()

if __name__ == "__main__":
    # cmd argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", action="store")
    parser.add_argument("-t", "--threads", type=int, action="store")
    parser.add_argument("--num_processes", type=int, action="store", default=1)
    parser.add_argument("-s", "--steps", type=int, action="store")
    parser.add_argument("-b", "--batch_size", default=1, type=int, action="store")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-i", "--iteration", type=int, default=1)
    parser.add_argument("--log_path", action="store")
    args = parser.parse_args()

    mp.set_start_method("spawn")
    inst = Inference(args)
    inst.load_models()

    # inst.run()

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=run_process, args=(inst,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    #for i in range(args.iteration):
    #with torch.mps.profiler.profile(mode="interval", wait_until_completed=True):
