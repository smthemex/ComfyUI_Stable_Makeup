# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import folder_paths
import os
import torch
#from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
import diffusers
from transformers import CLIPTextModel

dif_version = str(diffusers.__version__)
dif_version_int = int(dif_version.split(".")[1])

from diffusers import (DDIMScheduler, ControlNetModel,
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler,
                       DPMSolverSinglestepScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler,
                       DDPMScheduler, LCMScheduler, StableDiffusionPipeline, )
from .pipeline_sd15 import StableDiffusionControlNetPipeline
from .detail_encoder.encoder_plus import detail_encoder
from .import spiga_draw
from PIL import Image
from .facelib import FaceDetector
from comfy.utils import common_upscale
import numpy as np
import sys

makeup_current_path = os.path.dirname(os.path.abspath(__file__))
node_path_dir = os.path.dirname(makeup_current_path)
comfy_file_path = os.path.dirname(node_path_dir)

weigths_current_path = os.path.join(folder_paths.models_dir, "stable_makeup")

if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)

scheduler_list = ["DDIM",
    "Euler",
    "Euler a",
    "DDPM",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "DPM++ SDE",
    "DPM++ SDE Karras",
    "DPM2",
    "DPM2 Karras",
    "DPM2 a",
    "DPM2 a Karras",
    "Heun",
    "LCM",
    "LMS",
    "LMS Karras",
    "UniPC",
]

def get_sheduler(name):
    scheduler = False
    if name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "DDIM":
        scheduler = DDIMScheduler()
    elif name == "DDPM":
        scheduler = DDPMScheduler()
    elif name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "DPM++ 2M SDE":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ 2M SDE Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ SDE":
        scheduler = DPMSolverSinglestepScheduler()
    elif name == "DPM++ SDE Karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "DPM2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "DPM2 Karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "DPM2 a":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "DPM2 a Karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "Heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "LCM":
        scheduler = LCMScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    return scheduler

paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))

if paths:
    paths = ["none"] + [x for x in paths if x]
else:
    paths = ["none", ]


def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil

def get_local_path(comfy_file_path, model_path):
    path = os.path.join(comfy_file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path

def instance_path(path, repo):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(comfy_file_path, path)
            repo = get_instance_path(model_path)
    return repo


class StableMakeup_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "clip":("STRING", {"default": "openai/clip-vit-large-patch14"}),
                "scheduler": (scheduler_list,),
            }
        }

    RETURN_TYPES = ("MODEL","MODEL",)
    RETURN_NAMES = ("model","makeup_encoder",)
    FUNCTION = "main_loader"
    CATEGORY = "Stable_Makeup"

    def main_loader(self,ckpt_name,clip,scheduler):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        scheduler_used = get_sheduler(scheduler)
        makeup_encoder_path = os.path.join(weigths_current_path,"pytorch_model.bin")
        id_encoder_path = os.path.join(weigths_current_path,"pytorch_model_1.bin")
        pose_encoder_path = os.path.join(weigths_current_path,"pytorch_model_2.bin")
        original_config_file=os.path.join(folder_paths.models_dir,"configs","v1-inference.yaml")
        sd15_config="Lykon/dreamshaper-8"
        if dif_version_int >= 28:
             pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path,config=sd15_config, original_config=original_config_file)
        else:
            pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path, config=sd15_config,original_config_file=original_config_file)
        pipe.to("cuda")
        Unet= pipe.unet
        vae=pipe.vae
        text_encoder = pipe.text_encoder
        id_encoder = ControlNetModel.from_unet(Unet)
        pose_encoder = ControlNetModel.from_unet(Unet)
        makeup_encoder = detail_encoder(Unet, clip, "cuda", dtype=torch.float32)
        makeup_state_dict = torch.load(makeup_encoder_path)
        id_state_dict = torch.load(id_encoder_path)
        id_encoder.load_state_dict(id_state_dict, strict=False)
        pose_state_dict = torch.load(pose_encoder_path)
        pose_encoder.load_state_dict(pose_state_dict, strict=False)
        makeup_encoder.load_state_dict(makeup_state_dict, strict=False)
        id_encoder.to("cuda")
        pose_encoder.to("cuda")
        makeup_encoder.to("cuda")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/dreamshaper-8",
            safety_checker=None,
            unet=Unet,
            vae=vae,
            text_encoder=text_encoder,
            controlnet=[id_encoder, pose_encoder],
            torch_dtype=torch.float32).to("cuda")
        pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config)
        return (pipe,makeup_encoder)

class StableMakeup_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id_image": ("IMAGE",),
                "makeup_image": ("IMAGE",),
                "pipe": ("MODEL",),
                "makeup_encoder": ("MODEL",),
                "facedetector": (["mobilenet","resnet"],),
                "dataname": (["300wpublic", "300wprivate","merlrav","wflw"],),
                "cfg": ("FLOAT", {"default": 1.6, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "width": ("INT", {"default": 512, "min": 256, "max": 768, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 768, "step": 64, "display": "number"}),
                
               }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "makeup_main"
    CATEGORY = "Stable_Makeup"
    
    
    def makeup_main(self, id_image, makeup_image, pipe, makeup_encoder,facedetector,dataname,cfg, steps,
                      width, height ):
        
        if facedetector=="mobilenet":
            weight_path=os.path.join(weigths_current_path, "mobilenet0.25_Final.pth")
        else:
            weight_path=os.path.join(weigths_current_path, "resnet50.pth")
        detector = FaceDetector(name=facedetector,weight_path=weight_path)
        
        def get_draw(pil_img, size,dataname):
            spigas = spiga_draw.spiga_process(pil_img, detector,dataname)
            if spigas == False:
                width, height = pil_img.size
                black_image_pil = Image.new('RGB', (width, height), color=(0, 0, 0))
                return black_image_pil
            else:
                spigas_faces = spiga_draw.spiga_segmentation(spigas, size=size)
                return spigas_faces
            
        id_image=nomarl_upscale(id_image, width, height)
        makeup_image = nomarl_upscale(makeup_image, width, height)
        pose_image = get_draw(id_image, size=width,dataname=dataname)
        result_img = makeup_encoder.generate(id_image=[id_image, pose_image], makeup_image=makeup_image,num_inference_steps=steps,
                                             pipe=pipe, guidance_scale=cfg)
        
        image=pil2narry(result_img)
        return (image,)



NODE_CLASS_MAPPINGS = {
    "StableMakeup_LoadModel":StableMakeup_LoadModel,
    "StableMakeup_Sampler": StableMakeup_Sampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StableMakeup_LoadModel":"StableMakeup_LoadModel",
    "StableMakeup_Sampler": "StableMakeup_Sampler",
}
