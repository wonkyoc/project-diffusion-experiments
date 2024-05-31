#!/home/wonkyoc/miniconda3/bin/python

import torch
#import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, \
        PNDMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import make_image_grid
from diffusers.utils import logging

from tqdm.auto import tqdm
import time
import os
import argparse

TEST = 1
print("pid=", os.getpid())

# xzl: time measurement...write to a separate file
class Perf:
    def __init__(self, args):
        self.iteration = args.iteration
        self.total_cond_time = 0
        self.total_uncond_time = 0
        self.total_unet_time = 0
        self.total_denoise_time = 0
        self.total_vae_time = 0
        if TEST == 1:
            self.log_file = f"test-{args.device}-t{args.threads}-b{args.batch_size}-pid{os.getpid()}.log"
        else:
            self.log_file = f"{args.device}-t{args.threads}-b{args.batch_size}-pid{os.getpid()}.log"

    def save_log(self):
        avg_cond_time = self.total_cond_time / self.iteration
        avg_uncond_time = self.total_uncond_time / self.iteration
        avg_unet_time = self.total_unet_time / self.iteration
        avg_denoise_time = self.total_denoise_time / self.iteration
        avg_vae_time = self.total_vae_time / self.iteration
        with open(self.log_file, "w") as f:
            f.write(f"avg_cond_time (all steps)={avg_cond_time}\n")
            f.write(f"avg_uncond_time (all steps)={avg_uncond_time}\n")
            f.write(f"avg_unet_time (all steps)={avg_unet_time}\n")
            f.write(f"avg_denoise_time (per step)={avg_denoise_time}\n")
            f.write(f"avg_vae_time={avg_vae_time}")
        

def make_image(i):
    i = (i / 2 + 0.5).clamp(0, 1).squeeze()
    i = (i.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(i)

def run_inference(logger, args, perf):
    # Model
    model_id = "runwayml/stable-diffusion-v1-5"
    #model_id = "~/git/models/sd-v1-5.ckpt"
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", 
            use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet",
            use_safetensors=True)
    #scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Hyperparameters
    device = args.device
    batch_size = args.batch_size
    threads = args.threads
    prompt = args.prompt
    num_inference_steps = args.steps

    print(f"config: device={device} bs={batch_size} threads={threads} steps={num_inference_steps}")

    torch.set_num_threads(threads)      # xzl: is this useful at all?
    height, width = 512, 512
    #prompt = "a portrait of a woman with medium length black hair and a fringe, face close up, light blue eyeliner, wearing sports trousers and a sweatshot, dancing in a ballet studio, lensbaby"
    prompts = [prompt] * batch_size   # xzl: duplicate prompts.... by bs
    guidance_scale = 7.5
    generator = [torch.Generator(device).manual_seed(i) for i in range(batch_size)]
    filename = f"c{threads}-b{batch_size}"  # xzl: output filename

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # Tokenizer
    text_input = tokenizer(prompts, padding="max_length", 
            max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    perf.total_cond_time += time.time() - start

    # Unconditional text prompt
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")

    start = time.time()
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    perf.total_uncond_time += time.time() - start

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # create random latents
    shape = (1, unet.config.in_channels, height // 8, width // 8)
    latents = [
            torch.randn(shape, generator=generator[i], device=device)
            for i in range(batch_size)
            ]
    latents = torch.cat(latents, dim=0).to(device)
    latents = latents * scheduler.init_noise_sigma

    # xzl: latents shape (bs, ch, 64, 64)  height // 8, width // 8

    # set time step
    scheduler.set_timesteps(num_inference_steps)

    # denoise
#    with profile(activities=[ProfilerActivity.CPU],
#            profile_memory=True,
#            with_stack=True,
#            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./results/{filename}.trace'),
#            record_shapes=True) as prof:
    total_denoise_time = 0
    for t in tqdm(scheduler.timesteps):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        start = time.time()
        #with record_function("## U-Net ##"):
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        #logger.info(f"t={t} time={end - start}")
        total_denoise_time += time.time() - start

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        avg_denoise_time = total_denoise_time / len(scheduler.timesteps)
    perf.total_unet_time += total_denoise_time
    perf.total_denoise_time += avg_denoise_time

    #logger.info(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    #prof.export_memory_timeline(f"results/{filename}-memory.html", device="cpu")
    #prof.export_memory_timeline(f"results/{filename}-memory.raw.json.gz", device="cpu")
    #prof.export_memory_timeline(f"results/{filename}-memory.json.gz", device="cpu")

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents

    start = time.time()
    with torch.no_grad():
        decoded_images = vae.decode(latents).sample
    perf.total_vae_time += time.time() - start

    images = [make_image(i) for i in decoded_images]
    if batch_size == 1:
        images = make_image_grid(images, batch_size, batch_size)
    elif batch_size == 3:
        images = make_image_grid(images, 1, 3)
    else:
        rest = int(batch_size / 2)
        images = make_image_grid(images, 2, int(batch_size / 2))

    images.save(f"{filename}.png")

if __name__ == "__main__":
    # cmd argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", action="store")
    parser.add_argument("-t", "--threads", type=int, action="store")
    parser.add_argument("-s", "--steps", type=int, action="store")
    parser.add_argument("-b", "--batch_size", default=1, type=int, action="store")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-i", "--iteration", type=int, default=1)
    parser.add_argument("--log_path", action="store")
    args = parser.parse_args()


    # Debugging
    logging.set_verbosity_info()
    logger = logging.get_logger("diffusers")
    perf = Perf(args)

    # xzl: mps profiler. 
    for i in range(args.iteration):
        with torch.mps.profiler.profile(mode="interval", wait_until_completed=True):
            run_inference(logger, args, perf)
        time.sleep(5)
    
    perf.save_log()
