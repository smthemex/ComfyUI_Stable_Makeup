# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import folder_paths
import os
import torch


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

makeup_current_path = os.path.dirname(os.path.abspath(__file__))
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


class StableMakeup_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "clip":(folder_paths.get_filename_list("clip"),),
                "lora":(["none"]+folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "lora_trigger_words":("STRING", {"default": "best"}),
                "scheduler": (scheduler_list,),
            }
        }

    RETURN_TYPES = ("MAKEUP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "Stable_Makeup"

    def main_loader(self,ckpt_name,clip,lora,lora_scale,lora_trigger_words,scheduler):
        clip=folder_paths.get_full_path("clip", clip)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        scheduler_used = get_sheduler(scheduler)
        makeup_encoder_path = os.path.join(weigths_current_path,"pytorch_model.bin")
        id_encoder_path = os.path.join(weigths_current_path,"pytorch_model_1.bin")
        pose_encoder_path = os.path.join(weigths_current_path,"pytorch_model_2.bin")
        original_config_file=os.path.join(folder_paths.models_dir,"configs","v1-inference.yaml")
        sd15_config=os.path.join(makeup_current_path,"sd15_config")
        try:
             pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path,config=sd15_config, original_config=original_config_file)
        except:
            pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path, config=sd15_config,original_config_file=original_config_file)
        pipe.to("cuda")
        Unet= pipe.unet
        vae=pipe.vae
        text_encoder = pipe.text_encoder
        id_encoder = ControlNetModel.from_unet(Unet)
        pose_encoder = ControlNetModel.from_unet(Unet)
        repo=os.path.join(makeup_current_path,"clip")
        makeup_encoder = detail_encoder(Unet, clip,repo, "cuda", dtype=torch.float32)
        makeup_state_dict = torch.load(makeup_encoder_path)
        id_state_dict = torch.load(id_encoder_path)
        
        id_encoder.load_state_dict(id_state_dict, strict=False)
        pose_state_dict = torch.load(pose_encoder_path)
        pose_encoder.load_state_dict(pose_state_dict, strict=False)
        makeup_encoder.load_state_dict(makeup_state_dict, strict=False)
        
        del id_state_dict,makeup_state_dict,pose_state_dict
        torch.cuda.empty_cache()
        
        id_encoder.to("cuda")
        pose_encoder.to("cuda")
        makeup_encoder.to("cuda")
        if lora!="none":
            lora_path = folder_paths.get_full_path("loras", lora)
            pipe.load_lora_weights(lora_path, adapter_name=lora_trigger_words)
            pipe.fuse_lora(adapter_names=[lora_trigger_words, ], lora_scale=lora_scale)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd15_config,
            safety_checker=None,
            unet=Unet,
            vae=vae,
            text_encoder=text_encoder,
            controlnet=[id_encoder, pose_encoder],
            torch_dtype=torch.float32).to("cuda")
        pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config)
        
        return ({"pipe":pipe,"makeup_encoder":makeup_encoder},)

class StableMakeup_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id_image": ("IMAGE",),
                "makeup_image": ("IMAGE",),
                "model": ("MAKEUP_MODEL",),
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
    
    
    def makeup_main(self, id_image, makeup_image, model,facedetector,dataname,cfg, steps,width, height ):
        
        pipe=model.get("pipe")
        makeup_encoder=model.get("makeup_encoder")
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
    "StableMakeup_Sampler": "StableMakeup_Sampler"
}
