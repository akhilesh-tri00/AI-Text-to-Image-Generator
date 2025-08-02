# Project Name -> [ AI Text to Image Generator ]

# Project Description...
'''
This project is a desktop-based AI Image Generator built using Tkinter and Stable Diffusion. 
It allows users to input a text prompt, which is then used to generate high-quality images
using the Stable Diffusion v1.5 model from Hugging Face’s diffusers library. 
The app leverages PyTorch to run on either CPU or GPU, depending on availability, 
for efficient image generation. 
Users can view the generated image directly in the GUI and download it as a PNG file. 
The GUI is simple, interactive, and ideal for beginners exploring generative AI. 
This project combines deep learning with a user-friendly interface to bring text-to-image generation to the desktop.'''

# Project code:-

# importing required libraries
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline

# ---- Load Stable Diffusion pipeline only once ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",   # this is stable diffussion v1.5 model
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

# ---- Variable to hold the latest generated image ----
generated_image = None

# ---- Function to generate image ----
def generate_image():
    global generated_image

    prompt = prompt_entry.get().strip()
    if not prompt:
        messagebox.showerror("Input Error", "Please enter a prompt!")
        return

    try:
        # Generate image from prompt
        with torch.autocast(device if device == "cuda" else "cpu"):
            image = pipe(prompt, guidance_scale=7.5).images[0]

        generated_image = image  # Store for download

        # Resize for display
        image_resized = image.resize((512, 512))
        tk_image = ImageTk.PhotoImage(image_resized)

        # Show on canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image

    except Exception as e:
        messagebox.showerror("Error", f"Image generation failed:\n{e}")

# ---- Function to download image ----
def download_image():
    if generated_image is None:
        messagebox.showinfo("No Image", "Please generate an image first.")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
        title="Save Generated Image"
    )

    if file_path:
        try:
            generated_image.save(file_path)
            messagebox.showinfo("Success", f"Image saved at:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

# ---- Tkinter GUI ----
root = tk.Tk()
root.title("AI Text to Image Generator")

# Prompt Entry
tk.Label(root, text="Enter prompt:").pack()
prompt_entry = tk.Entry(root, width=60)
prompt_entry.pack()

# Buttons
tk.Button(root, text="Generate Image", command=generate_image).pack(pady=10)
tk.Button(root, text="Download Image", command=download_image).pack(pady=5)

# Canvas to display image
canvas = tk.Canvas(root, width=512, height=512, bg="white")
canvas.pack()

# Run the app
root.mainloop()
