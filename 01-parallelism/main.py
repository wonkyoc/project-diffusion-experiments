#!/home/wonkyoc/miniconda3/bin/python
import gc
import torch
import json
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

from multiprocessing.managers import BaseManager
from multiprocessing import Lock
from enum import Enum


class State(Enum):
    IDLE = 0
    RUNNING = 1

class ResourceManager():
    def __init__(self):
        self.cpu_state = State.IDLE # XXX: tbd
        self.gpu_state = State.IDLE
        self.events = []
        self.lock = Lock()
    
    def set_gpu_state(self, state):
        self.gpu_state = state

    def get_gpu_state(self):
        return self.gpu_state

    # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.yr4qxyxotyw
    def add_event(self, name: str, cat, ph: str, ts: int, pid: int, args):
        self.events.append({
            "name": name,
            "cat": cat,
            "ph": ph,
            "ts": ts * 1000000, # tracing supports microsec
            "pid": pid,
            "tid": pid,
            "args": args
        })

    def get_events(self):
        return self.events

    def acquire_lock(self):
        return self.lock.acquire(block=False)

    def release_lock(self):
        return self.lock.release()



class Inference():
    def __init__(self, args, inst_id):
        self.id = inst_id
        self.pid = os.getpid()
        self.debug = True

        # Hyperparameters
        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size
        self.threads = args.threads
        self.prompt = args.prompt   # single prompt
        self.num_inference_steps = args.steps
        self.height, self.width = 512, 512
        self.guidance_scale = 7.5
        self.model_id = args.model

        # logging
        self.log_dir = args.log_dir
        self.total_cond_time = 0
        self.total_uncond_time = 0
        self.total_unet_time = 0
        self.total_denoise_time = 0
        self.total_vae_time = 0
        self.copy_time = 0

        # pipeline
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.scheduler = None

        # torch setting
        torch.set_num_threads(self.threads)

    # Chrome tracing format


    def load_models(self):
        # Model
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", use_safetensors=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder", 
                use_safetensors=True)
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet",
                use_safetensors=True)
        #scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        if self.device == "cpu":
            self.unet.share_memory()
            self.vae.share_memory()
            self.text_encoder.share_memory()

    def _copy_data(self, device):
        self.device = device
        self.text_encoder.to(device)
        self.unet.to(device)
        self.vae.to(device)

    def run(self, ctx):
        gc.collect()
        ctx.add_event("run", f"instance-{self.id}", "B", time.time(), self.pid, {})

        if self.device == "mps":
            self.batch_size = self.args.gpu_batch_size
            self._copy_data(self.device)

        self.prompts = [self.prompt] * self.batch_size
        
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
            if self.args.offload == True:
                if ctx.get_gpu_state() == State.IDLE and ctx.acquire_lock() == True:
                    ctx.add_event("switch", f"instance-{self.id}", "B", time.time(), self.pid, {})
                    print(f"cpu-{self.id} -> gpu-{self.id}")

                    # move data/model to device
                    start = time.time()
                    self._copy_data("mps")
                    latents = latents.to("mps")
                    text_embeddings = text_embeddings.to("mps")
                    self.copy_time = time.time() - start

                    # change status
                    ctx.set_gpu_state(State.RUNNING)
                    ctx.add_event("switch", f"instance-{self.id}", "E", time.time(), self.pid, {})

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
        ctx.add_event("run", f"instance-{self.id}", "E", time.time(), self.pid, {})

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

        images.save(f"{self.log_dir}/{self.image_name}.png")
        gc.collect()

    def save_log(self):
        if self.debug == True:
            self.log_name = f"{self.log_dir}/debug-pid{os.getpid()}-{self.device}-t{self.threads}-b{self.batch_size}-p{self.num_instances}"
        else:
            self.log_name = f"{self.log_dir}/pid{os.getpid()}-{self.device}-t{self.threads}-b{self.batch_size}-p{self.num_instance}"

        avg_cond_time = self.total_cond_time / self.args.iteration
        avg_uncond_time = self.total_uncond_time / self.args.iteration
        avg_unet_time = self.total_unet_time / self.args.iteration
        avg_denoise_time = self.total_denoise_time / self.args.iteration
        avg_vae_time = self.total_vae_time / self.args.iteration
        
        with open(f"{self.log_name}.log", "w") as f:
            f.write(f"avg_cond_time (all steps)={avg_cond_time}\n")
            f.write(f"avg_uncond_time (all steps)={avg_uncond_time}\n")
            f.write(f"avg_unet_time (all steps)={avg_unet_time}\n")
            f.write(f"avg_denoise_time (per step)={avg_denoise_time}\n")
            f.write(f"avg_vae_time={avg_vae_time}\n")
            f.write(f"copy_time={self.copy_time}\n")

def run_process(inst, shared_inst):
    print(f"inst-{inst.id} instance ({inst.device}) runs")

    if inst.device == "mps" and shared_inst.acquire_lock():
        shared_inst.set_gpu_state(State.RUNNING)

    inst.pid = os.getpid()
    inst.run(shared_inst)

    if inst.device == "mps":
        shared_inst.set_gpu_state(State.IDLE)
        shared_inst.release_lock()

    print(f"inst-{inst.id} instance ({inst.device}) terminates")

if __name__ == "__main__":
    # cmd argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", action="store")
    parser.add_argument("-t", "--threads", type=int, action="store")
    parser.add_argument("-s", "--steps", type=int, action="store")
    parser.add_argument("-b", "--batch_size", default=1, type=int, action="store")
    parser.add_argument("--gpu_batch_size", default=1, type=int, action="store")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-i", "--iteration", type=int, default=1)
    parser.add_argument("--log_dir", action="store")
    parser.add_argument("--num_cpu_instances", type=int, action="store")
    parser.add_argument("--num_gpu_instances", type=int, action="store", default=1)
    parser.add_argument("--model", action="store")
    parser.add_argument("--offload", action="store_true")

    args = parser.parse_args()

    mp.set_start_method("spawn")

    # Process manager
    BaseManager.register("ResourceManager", ResourceManager)
    manager = BaseManager()
    manager.start()
    shared_inst = manager.ResourceManager()
   
    # Inference instance
    inst = Inference(args=args, inst_id=-1)
    inst.load_models()

    processes = []
    inst.num_instances = num_instances = args.num_cpu_instances + args.num_gpu_instances
    num_images = (args.num_cpu_instances * args.batch_size) + (args.num_gpu_instances * args.gpu_batch_size)

    # audit config
    with open(f"{args.log_dir}/config", "w") as f:
        f.write(f"bs={args.batch_size} gbs={args.gpu_batch_size} threads={args.threads} " +
        f"steps={args.steps} num_instances={num_instances} num_images={num_images} " + 
        f"model={args.model} " + f"num_cpu_isntances={args.num_cpu_instances} num_gpu_instances={args.num_gpu_instances} ")

    # counting inst
    cur_id = 0

    # generate gpu instances
    pid = os.getpid()
    shared_inst.add_event("main", "manager", "B", time.time(), pid, {})
    shared_inst.add_event("create_instance", "manager", "B", time.time(), pid, {})
    assert args.num_gpu_instances <= 1, f"num_gpu_instance [{args.num_gpu_instances}]: a single instance is enough"
    for rank in range(args.num_gpu_instances):
        inst.device = "mps"
        inst.id = cur_id
        p = mp.Process(target=run_process, args=(inst, shared_inst,))
        p.start()
        cur_id += 1
        processes.append(p)

    # generate cpu instances
    for rank in range(args.num_cpu_instances):
        inst.device = "cpu"
        inst.id = cur_id
        p = mp.Process(target=run_process, args=(inst, shared_inst, ))
        p.start()
        cur_id += 1
        processes.append(p)
    shared_inst.add_event("create_instance", "manager", "E", time.time(), pid, {})

    # wait
    for p in processes:
        p.join()
    shared_inst.add_event("main", "manager", "E", time.time(), pid, {})

    # logging
    with open(f"{args.log_dir}/events.json", "w") as f:
        json.dump(shared_inst.get_events(), f)


    #with torch.mps.profiler.profile(mode="interval", wait_until_completed=True):
