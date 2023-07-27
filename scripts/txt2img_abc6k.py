import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json

from diffusers import StableDiffusionPipeline

from argparse import Namespace
from copy import deepcopy

def preprocess_prompts(prompts):
    if isinstance(prompts, (list, tuple)):
        return [p.lower().strip().strip(".").strip() for p in prompts]
    elif isinstance(prompts, str):
        return prompts.lower().strip().strip(".").strip()
    else:
        raise NotImplementedError


def main():
          
    OPT_DEFAULT = Namespace(
        C=4,
        H=512, W=512,
        conjunction=False,
        ddim_eta=0.0,
        ddim_steps=50,
        f=8,
        fixed_code=False,
        from_file=None,
        laion400m=False,
        n_iter=1,
        n_rows=0,
        n_samples=1,
        outdir='outputs/txt2img-samples',
        parser_type='constituency',
        precision='autocast',
        prompt='A red teddy bear in a christmas hat sitting next to a glass',
        save_attn_maps=False,
        scale=7.5,
        seed=42,
        skip_grid=False,
        skip_save=False
    )
    
    resolution_types = ['II']
    conjunctions = ['sd_1_5']
    with open("scripts/ABC-6K.txt", "r") as f:
        lines = f.readlines()
        CC_500 = [l.rstrip() for l in lines]
    with open("scripts/seeds_ABC-6K.json", "r") as f:
        seeds = json.load(f)
    seeds_and_prompts = {int(i): {'seeds': seeds[i]} for i in seeds.keys() if int(i) > 602}
    print("len(seeds_and_prompts)", len(seeds_and_prompts))
    for i in list(seeds_and_prompts.keys()):
        seeds_and_prompts[i]['prompt'] = CC_500[i]
    # for file in os.listdir('CC-500/base-l'):
    #     if file.endswith(".png"): 
    #         index, seed, _ = file.split('_')
    #         index = int(index)
    #         seeds_and_prompts[index]['prompt'] = CC_500[index]
    # for i in list(seeds_and_prompts.keys()):
    #     if len(seeds_and_prompts[i]['seeds']) == 0:
    #         seeds_and_prompts.pop(i)


    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    def generate(opt):
        seed_everything(opt.seed)
        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir
        prompt = preprocess_prompts( opt.prompt)
        assert prompt is not None
        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                img = pipe(prompt, num_inference_steps=opt.ddim_steps).images[0]  
                img.save(os.path.join(outpath, opt.outfname))
    
    for conj in conjunctions:
        for res_type in resolution_types:
            for index, prompt_and_seeds in seeds_and_prompts.items():
                    prompt = prompt_and_seeds['prompt']
                    seeds = prompt_and_seeds['seeds']
                    for seed in seeds:
                        opt = deepcopy(OPT_DEFAULT)
                        if conj == 'standard':
                            opt.conjunction = False
                        elif conj == 'special':
                            opt.conjuction = True
                        if res_type == 'II':
                            opt.W = 512
                            opt.H = 512
                        elif res_type == 'I':
                            opt.W = 64
                            opt.H = 64
                        opt.seed = seed
                        opt.prompt = prompt
                        opt.outdir = f'outputs_abc6k/{conj}'
                        opt.outfname = f'{index}_{seed}_{res_type}.png'
                        print("opt", opt)
                        generate(opt)
                        print(f"generated {opt.outfname} (prompt: {opt.prompt})")


if __name__ == "__main__":
    main()
