# app.py
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 50
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (800, 800)
    image_gen_guidance_scale = 12
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 12

auth_token = "YOUR HUGGING FACE TOKEN HERE"
modelid = "CompVis/stable-diffusion-v1-4"

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

def main():
    st.title("Stable Diffusion Image Generator")

    # Load the first model
    image_gen_model_1 = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id, torch_dtype=torch.float16,
        revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
    )
    image_gen_model_1 = image_gen_model_1.to(CFG.device)

    # Load the second model
    image_gen_model_2 = StableDiffusionPipeline.from_pretrained(
        modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token
    )
    image_gen_model_2 = image_gen_model_2.to(CFG.device)

    prompt = st.text_input("Enter prompt:", "beautiful sunset")

    if st.button("Generate Images"):
        with st.spinner("Generating images..."):
            image_1 = generate_image(prompt, image_gen_model_1)
            image_2 = generate_image(prompt, image_gen_model_2)

        st.image([image_1, image_2], caption=["Model 1 Image", "Model 2 Image"], use_column_width=True)

# if __name__ == "__main__":
#     main()
