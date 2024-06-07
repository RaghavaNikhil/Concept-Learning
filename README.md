# NLP-fall-23-concept-edit-learning

Consider a text-to-image (T2I) generative model $g$.
This takes text prompt $y$ as input and generates image $x_{gen} = g(y)$.

Concept is very broad term. What is concept? Is it some object? Is it the properties of the objects?
Or is it both? What about abstract concepts? Similarly, everything can be categorized as concepts.

Now, given a concept image $x$ ("a photo of goofy"), we want to edit the image and get $x_e$ ("a photo of goofy in neon lights").
We can achieve this using simple diffusion process to combine both concept image and text prompt (example shown in [sample_test.py](src/sample_test.py) script).

However, this leads to the similar image with neon light, but it's not EXACTLY the same original image.
So, how can we achieve this?
Another way to solve this might be use the two reference images as example to learn the edit direction.
Then apply it to the new target image (example shown in last half in [sample_test.py](src/sample_test.py) script).
However, this still leads to good but unstructured image.

Therefore, the goal of this project is to introduce new conditional module that takes reference images (or edit direction) along with text prompt to generate accurate image embedding.
This image embedding will be used by decoder to generate the image.

# Reading List

* CLIP model paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* MagicBrush dataset: [MAGICBRUSH: A Manually Annotated Dataset for Instruction-Guided Image Editing](https://arxiv.org/abs/2306.10012)
* Latest text-conditioned image editing paper: [Visual Instruction Inversion: Image Editing via Visual Prompting](https://arxiv.org/abs/2307.14331)

### Notes: 
* It is important to understand the CLIP model as it is the core of our method.
* Understand of the MagicBrush dataset is important as we will use it to train the model.
* Last image editing paper is for reference. Understand the motivation, problem statement, and how they evaluate their model. (**it's fine if you don't understand the complex modeling strategy mentioned in the paper**)

# Project Goals

* (2 weeks) Play with the [sample_test.py](src/sample_test.py) script and try to play with different settings on MagicBrush + Visual Instruction examples. Here, we want to observe how much performance do we get without training.
  * Spend some time to learn the CLIP model and how it is used inside the Kandinsky model within the `sample_test.py`.
* (2 weeks) Prepare the dataset and create first version of the training procedure. **[more details will be provided soon]**
* (2 weeks) Introduce different loss functions to improve the results and get the SotA method. 
* (1 week) Summarize the findings and write the report.
* If interested and results are promising then extend this work and write the research paper.
