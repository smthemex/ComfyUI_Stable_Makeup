# ComfyUI_Stable_Makeup
You can apply makeup to the characters when use ComfyUI

Stable_Makeup  From: [Stable_Makeup](https://github.com/Xiaojiu-z/Stable-Makeup)

Update
---
**2025/01/10**

* add lora support,you can try 4 step lora or other； 
* 增加加速Lora或者常规Lora的支持； 


1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Stable_Makeup.git
```  
  
2.requirements  
----
only insightface in requirements.txt   

```
pip install -r requirements.txt
```  
按理是不需要装特别的库，因为内置了，如果还是库丢失，请单独安装.
便携包和秋叶包请注意使用python -m pip install 

If the module is missing, please open "no need requirements.txt" , pip install or python -m pip install 
  missing module.    

3 Need  model 
----
模型的下载地址比较杂，所以使用前请下下载，并存放在ComfyUI/models/stable_makeup 文件夹下：   
The download address for the model is quite miscellaneous, so please download it before use and store it in the ComfyUI/models/table_makeup folder:  

3.1  spiga_300wpublic.pt or other models  [link](https://huggingface.co/aprados/spiga/tree/main)   

3.2  pytorch_model.bin  
     pytorch_model_1.bin  
     pytorch_model_2.bin   [link](https://drive.google.com/drive/folders/1397t27GrUyLPnj17qVpKWGwg93EcaFfg)

3.3  mobilenet0.25_Final.pth [link](https://drive.google.com/uc?export=download&id=1G3VsfgiQb16VyFnOwEVDgm2g8-9qN0-9)    
     or     
     resnet50.pth    [link](https://www.dropbox.com/s/8sxkgc9voel6ost/resnet50.pth?dl=1)  
       
3.4  clip模型，外置为输入格式,可以引导至本地其他路径。  
    "openai/clip-vit-large-patch14" clip models

3.5 SD1.5  
     any sd1.5 weights,   

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
├── ComfyUI/models/checkpoints
|         ├──  any sd1.5 weights,
```
首次使用需要下载openai/clip-vit-large-patch14


Example
-----
 
 ![](https://github.com/smthemex/ComfyUI_Stable_Makeup/blob/main/example.png)


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

