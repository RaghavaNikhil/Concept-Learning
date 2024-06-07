from diffusers import KandinskyV22Pipeline, KandinskyV22PriorEmb2EmbPipeline, KandinskyV22PriorPipeline
from diffusers.utils import load_image
import torch

pipe_prior = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
)
pipe_prior.to("cuda:1")  # set cuda device

pipe = KandinskyV22Pipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
)
pipe.to("cuda")


"""Add the neon light into existing dog photo
"""

# prompt = "neon lights"  # add edit instruction
# img = load_image(
#     "./assets/goofy.png"  # provide the path to the image
# )
# image_emb, negative_image_emb = pipe_prior(prompt, image=img, strength=0.7).to_tuple()
#
# image = pipe(
#     image_embeds=image_emb,
#     negative_image_embeds=negative_image_emb,
#     height=768,
#     width=768,
#     num_inference_steps=100,
# ).images
#
# image[0].save("./data/tests/test.png")


"""Find the edit direction using goofy examples and edit different image
"""

prompt = ""
source_img1 = load_image(
    "./assets/goofy.png"  # provide the path to the image
)
source_img2 = load_image(
    "./assets/goofy_w_neonlight.png"  # provide the path to the image
)

source_img1_feats = pipe_prior._encode_image([source_img1], device="cuda:1", num_images_per_prompt=1)
source_img2_feats = pipe_prior._encode_image([source_img2], device="cuda:1", num_images_per_prompt=1)

# get the edit direction --> edit = feat(image2) - feat(image1)
edit_direction = source_img2_feats - source_img1_feats

target_img = load_image(
    "./assets/dog.jpg"  # provide the path to the image
)

# ideally this strength should be 0.0
target_image_emb, target_negative_image_emb = pipe_prior(prompt, image=target_img, strength=0.1).to_tuple()

# apply the previously calculated edit direction --> target = target + edit
target_image_emb = target_image_emb + edit_direction

image = pipe(
    image_embeds=target_image_emb,
    negative_image_embeds=target_negative_image_emb,
    height=768,
    width=768,
    num_inference_steps=100,
).images

image[0].save("./data/tests/test.png")
