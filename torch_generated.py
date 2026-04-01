import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import PIL.Image as InterpolationMissingOptionError



device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1").to(device)


def generate_new_image(prompt):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

prompt = "an acoustic guitar"
generated_image = generate_new_image(prompt)
output_path = "torch_generated_image.png"
generated_image.save(output_path)

generated_image.show()
