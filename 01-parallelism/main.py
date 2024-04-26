import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, \
        PNDMScheduler, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
from diffusers.utils import logging

from tqdm.auto import tqdm
import time


def make_image(i):
    i = (i / 2 + 0.5).clamp(0, 1).squeeze()
    i = (i.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(i)

# Debugging
logging.set_verbosity_info()
logger = logging.get_logger("diffusers")

# Model
model_id = "runwayml/stable-diffusion-v1-5"
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", 
        use_safetensors=True)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet",
        use_safetensors=True)
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

# Hyperparameters
device = "cpu"
batch_size = 1
core = 24
height, width = 512, 512
prompt = "a portrait of a woman with medium length black hair and a fringe, face close up, light blue eyeliner, wearing sports trousers and a sweatshot, dancing in a ballet studio, lensbaby"
prompts = [prompt] * batch_size
num_inference_steps = 1
guidance_scale = 7.5
generator = [torch.Generator("cpu").manual_seed(i) for i in range(batch_size)]
filename = f"c{core}-b{batch_size}"

torch.set_num_threads(core)

# Tokenizer
text_input = tokenizer(prompts, padding="max_length", 
        max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

# Unconditional text prompt
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# create random latents
shape = (1, unet.config.in_channels, height // 8, width // 8)
latents = [
        torch.randn(shape, generator=generator[i], device=device)
        for i in range(batch_size)
        ]
latents = torch.cat(latents, dim=0).to(device)
latents = latents * scheduler.init_noise_sigma

# set time step
scheduler.set_timesteps(num_inference_steps)

# denoise
total = 0
with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./{filename}.trace'),
        record_shapes=True) as prof:
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        start = time.time()
        with record_function("## U-Net ##"):
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        end = time.time()
        #logger.info(f"t={t} time={end - start}")
        total += end - start

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        # scale and decode the image latents with vae

logger.info(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
prof.export_memory_timeline(f"{filename}-memory.html", device="cpu")
prof.export_memory_timeline(f"{filename}-memory.raw.json.gz", device="cpu")
prof.export_memory_timeline(f"{filename}-memory.json.gz", device="cpu")
logger.info(f"avg_time={total/len(scheduler.timesteps)}")

latents = 1 / 0.18215 * latents

torch.set_num_threads(24)
start = time.time()
with torch.no_grad():
    decoded_images = vae.decode(latents).sample
end = time.time()

print(f"vae {end - start}")


images = [make_image(i) for i in decoded_images]
if batch_size == 1:
    images = make_image_grid(images, batch_size, batch_size)
else:
    rest = int(batch_size / 2)
    images = make_image_grid(images, 2, int(batch_size / 2))

images.save(f"{filename}.png")


