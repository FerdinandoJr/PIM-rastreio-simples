#https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/#:~:text=OpenCV%2DPython%20is%20a%20library,conversion%20methods%20available%20in%20OpenCV.
#https://stackoverflow.com/questions/46236180/opencv-imshow-will-cause-jupyter-notebook-crash
#https://docs.opencv.org/4.5.2/d4/dc6/tutorial_py_template_matching.html
#https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python

from google.colab import drive
drive.mount('/content/drive')

import cv2
import os
import imageio
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
from tabulate import tabulate


#Informações básicas
path = "/content/drive/MyDrive/PIM/TAREFA FINAL/gatos/"
name_base_file = "gatos"

def gerar_gif(img_name, imgs, p):
  imageio.mimsave(p+"/"+img_name+".gif", imgs)

def mostrarImagem(img):
  plt.subplot(122),plt.imshow(img,cmap = 'gray')
  plt.title('Imagem'), plt.xticks([]), plt.yticks([])
  plt.suptitle(cv2.TM_CCOEFF)
  plt.show()

def criarPasta(name):
  try:
    os.makedirs(name)
  except OSError:
    print("Pasta: "+name+" Já Existe!")

template = cv2.imread(path+ 'templateGato.png',0)
cv2_imshow(template)

#(Método, Nome da Pasta)
def converter(meth, folder_exit):
  global path, name_base_file
  template = cv2.imread(path+ 'templateGato.png',0)
  images_gif_img = []
  images_gif_res = []
  valores  = []
  w, h = template.shape[::-1]
  path_full = path+""+folder_exit
  
  criarPasta(path_full)
  criarPasta(path_full+"/res")
  criarPasta(path_full+"/img")
    
  onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
  
  for i in range(1, len(onlyfiles)):
    src = path+name_base_file+'{:0>5}'.format(i)+".png"
    img2 = cv2.imread(src)
    img = img2.copy()
    method = eval(meth)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converte a imagem para tons de cinza
        
    res = cv2.matchTemplate(gray, template,method)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(gray,top_left, bottom_right, 255, 2) #Encontra o Objeto na imagem e marca com um retângulo
    #cv2_imshow(gray)
    
    img_name = folder_exit+'{:0>5}'.format(i)+".png"

    #mostrarImagem(res)
    #mostrarImagem(img)

    cv2.imwrite(path_full+"/res/"+img_name, res)
    cv2.imwrite(path_full+"/img/"+img_name, gray) #Salva a imagem com a marcação
    
    images_gif_res.append(imageio.imread(path_full+"/res/"+img_name))
    images_gif_img.append(imageio.imread(path_full+"/img/"+img_name))
    
    valores.append([img_name, min_val, max_val])    
    print(name_base_file+'{:0>5}'.format(i)+".png"+" GEROU!")
  
  arquivo = open(path_full + '/valores.txt','w')
  arquivo.write(tabulate(valores, headers=["Name", "Valor Min", "Valor Max"])) #Armazena os valores máximos e minimos
  arquivo.close()
  
  gerar_gif("video_res", images_gif_res, path_full )
  gerar_gif("video_img", images_gif_img, path_full)

  
converter('cv2.TM_CCOEFF', 'TM_CCOEFF')
converter('cv2.TM_CCOEFF_NORMED', 'TM_CCOEFF_NORMED')
converter('cv2.TM_CCORR', 'TM_CCORR')
converter('cv2.TM_CCORR_NORMED', 'TM_CCORR_NORMED')
converter('cv2.TM_SQDIFF', 'TM_SQDIFF')
converter('cv2.TM_SQDIFF_NORMED', 'TM_SQDIFF_NORMED')
