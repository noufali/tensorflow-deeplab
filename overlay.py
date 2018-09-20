import os
from option import Options

import numpy as np
from PIL import Image

style_path = "/home/paperspace/PyTorch-Multi-Style-Transfer/experiments/stylized/"
overlay_path = "/home/paperspace/PyTorch-Multi-Style-Transfer/experiments/stylized (pre)/"
output_path = "/home/paperspace/PyTorch-Multi-Style-Transfer/experiments/output/"

files_style = os.listdir(style_path)
files_overlay = os.listdir(overlay_path)

files_style.sort(key=lambda f: int(''.join(filter(str.isdigit,f))))
files_overlay.sort(key=lambda f: int(''.join(filter(str.isdigit,f))))

i = 0

path = "/home/paperspace/PyTorch-Multi-Style-Transfer/experiments/"
style = Image.open(path + "1.jpg").convert("RGBA")
overlay = Image.open(path + "video.00001_mask.png").convert("RGBA")
mask = Image.open(path + "mask.png").convert("RGBA")

Image.composite(style,overlay,mask).save(path + "composite.png")



#while i < len(files_style):
  #print (files_style[i])
  #print (files_overlay[i])
  #style = Image.open(style_path + str(files_style[i])).convert("RGBA")
  #man = Image.open(overlay_path + str(files_overlay[i])).convert("RGBA")
  #style.paste(man,(0,0),man)
  #Image.blend(man,style,0.8).save(output_path + str(i) + ".png")
  #style.save(output_path + str(i) + ".png")
  #i += 1

