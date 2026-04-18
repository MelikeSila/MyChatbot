import time
import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image


def generate_image(pipe, prompt, output_path="generated_image.png"):
    """Generate image from text prompt"""
    image = pipe(
        prompt,
        num_inference_steps=50,  # More steps = better quality but slower
        guidance_scale=7.5,  # How closely to follow the prompt
    ).images[0]

    # Save the image
    image.save(output_path)
    print(f"Image saved to: {output_path}")

    # Display the image (if running locally with GUI)
    image.show()

    return image, output_path


def generate_image_with_time(pipe, prompt, output_path="generated_image.png"):
    start_time = time.time()
    image, path = generate_image(pipe, prompt, output_path)
    end_time = time.time()
    response_time = end_time - start_time
    return image, path, response_time


def image_generation_chatbot(pipe):
    print("Welcome to Image Generation Chatbot! Type 'exit' to quit.")
    print("Describe what image you want to generate.\n")

    image_counter = 0

    while True:
        prompt = input("You (describe image): ")
        if prompt.lower() == "exit":
            break

        image_counter += 1
        output_path = f"generated_image_{image_counter}.png"

        print(f"Generating image... (this may take 30-60 seconds)")
        image, path, response_time = generate_image_with_time(pipe, prompt, output_path)

        print(f"✓ Image generated successfully!")
        print(f"Saved to: {path}")
        print(f"Generation Time: {response_time:.2f} seconds\n")


if __name__ == "__main__":
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Choose your model
    # Option 1: Stable Diffusion 2.1 (good balance)
    model_name = "stabilityai/stable-diffusion-2-1"

    # Option 2: Stable Diffusion XL (better quality, slower)
    # model_name = "stabilityai/stable-diffusion-xl-base-1.0"

    # Option 3: Stable Diffusion 1.5 (faster, lighter)
    # model_name = "runwayml/stable-diffusion-v1-5"

    print(f"Loading {model_name}...")

    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for speed (requires GPU)
        token=token
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Using GPU")
    elif torch.backends.mps.is_available():  # Apple Silicon
        pipe = pipe.to("mps")
        print("Using Apple Silicon GPU")
    else:
        print("Using CPU (will be slow)")

    # Enable memory optimizations
    pipe.enable_attention_slicing()

    print("Model loaded! Ready to generate images.\n")
    image_generation_chatbot(pipe)