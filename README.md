# ComfyUI_Stable_Makeup
You can apply makeup to the characters in comfyui

Stable_Makeup  From: [Stable_Makeup](https://github.com/Xiaojiu-z/Stable-Makeup)

My ComfyUI node list：
-----
1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  

Update
---
---社区模型没有必要,之前添加是为了别的功能,目前无法实现,所以已经剔除;   
---可以尝试不同的数据集,当然,意味着你要多下载几个SPIGA模型;  
---修复没有预下载的模型时,无法加载的错误;  
---fix bug ,The community model is not necessary  
--- You can try different datasets, of course, which means you need to download a few more SPIGA models;  
---Fix the error where models that were not downloaded in advance cannot be loaded;

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Stable_Makeup.git
```  
  
2.requirements  
----
按理是不需要装特别的库，因为内置了，如果还是库丢失，请单独安装。或者打开no need requirements.txt，查看缺失的库是否在里面。  
秋叶包因为是沙箱模式，所以把缺失的库安装在系统的python库里。   
官方的整合包，用pip install AAA --target="BBB/python_embeded/Lib/site-packages" AAA替换为缺失的库，BBB是本地路径。   

If the module is missing, please open "no need requirements.txt" , pip install  missing module.    

3 Need  model 
----
模型的下载地址比较杂，所以使用前请下下载，并存放在ComfyUI/models/stable_makeup 文件夹下：   
The download address for the model is quite miscellaneous, so please download it before use and store it in the ComfyUI/models/table_makeup folder:  

3.1  spiga_300wpublic.pt or other models  [link](https://huggingface.co/aprados/spiga/tree/main)   

3.2  "pytorch_model.bin  & pytorch_model_1.bin  &  pytorch_model_2.bin"   [link](https://drive.google.com/drive/folders/1397t27GrUyLPnj17qVpKWGwg93EcaFfg)

3.3  mobilenet0.25_Final.pth [link](https://drive.google.com/uc?export=download&id=1G3VsfgiQb16VyFnOwEVDgm2g8-9qN0-9)    
     or     
     resnet50.pth    [link](https://www.dropbox.com/s/8sxkgc9voel6ost/resnet50.pth?dl=1)  
     
3.4   sd1.5的标准模型只需要以下结构内的文件，你如果直接用内置的repo下载，都会缓存到c盘，大概3.97G。   
     runwayml/stable-diffusion-v1-5 ,need unet models and vae encoder, You can check the following list, which includes the model and file locations，all 3.97G.     
    
3.5  clip模型，这个迟点看能否用comfyUI内置的，以及外置为输入格式,可以引导至本地其他路径。  
    "openai/clip-vit-large-patch14" clip models

Models list    
-----
```
├── ComfyUI/models/  
|     ├──stable_makeup
|         ├── mobilenet0.25_Final.pth
|         ├── pytorch_model.bin
|         ├── pytorch_model_1.bin
|         ├── pytorch_model_2.bin
|         ├── spiga_300wpublic.pt
|         ├── resnet50.pth
```
“runwayml/stable-diffusion-v1-5”只需要以下的文件，其他的不用下载，如果用默认的repo下载，会自动缓存到C盘  
"Runwayml/stable diffusion-v1-5 "only requires the following files, the rest do not need to be downloaded. If downloaded using the default repo, it will be automatically cached on the C drive   
```
├── ComfyUI/models/
|     ├──diffusers
|         ├── runwayml/stable-diffusion-v1-5
|             ├──unet
|                 ├── diffusion_pytorch_model.safetensors
|                 ├── config.json
|             ├──vae
|                 ├── diffusion_pytorch_model.safetensors
|                 ├── config.json
|             ├──tokenizer
|                 ├── merges.txt
|                 ├── special_tokens_map.json
|                 ├── tokenizer_config.json
|                 ├── vocab.json
|             ├──text_encoder
|                 ├── model.safetensors
|                 ├── config.json
|             ├──scheduler
|                 ├── scheduler_config.json
|             ├──safety_checker
|                 ├── config.json
|             ├──feature_extractor
|                 ├── preprocessor_config.json
|             ├──model_index.json
```

Example
-----
 
 ![](https://github.com/smthemex/ComfyUI_Stable_Makeup/blob/main/example/example.png)


6 Citation
------

``` python  
@article{zhang2024stable,
  title={Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model},
  author={Zhang, Yuxuan and Wei, Lifu and Zhang, Qing and Song, Yiren and Liu, Jiaming and Li, Huaxia and Tang, Xu and Hu, Yao and Zhao, Haibo},
  journal={arXiv preprint arXiv:2403.07764},
  year={2024}
}
```
SPIGA  From: [SPIGA](https://github.com/andresprados/SPIGA)
``` python  
@inproceedings{Prados-Torreblanca_2022_BMVC,
  author    = {Andrés  Prados-Torreblanca and José M Buenaposada and Luis Baumela},
  title     = {Shape Preserving Facial Landmarks with Graph Attention Networks},
  booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
  publisher = {{BMVA} Press},
  year      = {2022},
  url       = {https://bmvc2022.mpi-inf.mpg.de/0155.pdf}
}
```
FaceLib  From: [FaceLib](https://github.com/sajjjadayobi/FaceLib)

