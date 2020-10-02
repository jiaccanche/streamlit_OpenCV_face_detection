# Core Pkgs
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

@st.cache
def load_image(img):
  im = Image.open(img)
  return im

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')
def detect_face(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img_matrix = cv2.cvtColor(new_img,1)
  gray = cv2.cvtColor(img_matrix,cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray,1.1,4)
  for (x,y,w,h) in faces:
    cv2.rectangle(img_matrix,(x,y),(x+w,y+h),(255,0,0),2)
  return faces,img_matrix

def detect_eye(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img_matrix = cv2.cvtColor(new_img,1)
  gray = cv2.cvtColor(img_matrix,cv2.COLOR_BGR2GRAY)
  eyes = eye_cascade.detectMultiScale(gray,1.3,5)
  for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(img_matrix,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  return eyes,img_matrix

def detect_smile(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img_matrix = cv2.cvtColor(new_img,1)
  gray = cv2.cvtColor(img_matrix,cv2.COLOR_BGR2GRAY)
  smiles = eye_cascade.detectMultiScale(gray,1.1,4)
  for (sx,sy,sw,sh) in smiles:
    cv2.rectangle(img_matrix,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
  return smiles,img_matrix

def cannize_image(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img = cv2.cvtColor(new_img,1)
  img = cv2.GaussianBlur(img, (11, 11), 0)
  canny = cv2.Canny(img, 100, 150)
  return canny

def cartonize_image(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img_matrix = cv2.cvtColor(new_img,1)
  gray = cv2.cvtColor(img_matrix,cv2.COLOR_BGR2GRAY)
  #bordes
  gray = cv2.medianBlur(gray,5)
  edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9,9)
  #color
  color = cv2.bilateralFilter(img_matrix,9,300,300)
  #Cartoon
  cartoon = cv2.bitwise_and(color,color,mask=edges)
  return cartoon
  

def option_contrast(st):
  c_rate = st.sidebar.slider("Contraste", 0.5,3.5)
  return c_rate

def main():
  
  st.title("Aplicación de detección de rostros")
  st.text("Construido con Streamlit y OpenCv")
  st.set_option('deprecation.showfileUploaderEncoding', False)
  activities = ["Detección","Acerca"]
  choice = st.sidebar.selectbox("Seleccionar actividad", activities)
  
  if choice == 'Detección':
    st.subheader("Face Detection")
    image_file = st.file_uploader("Upload Image",type= ['jpg','png','jpeg'])
    if image_file is not None:
      our_image = Image.open(image_file)
      st.title("Imagen original")
      st.image(our_image, use_column_width=True)
    
       #Face detection
   
    task = ["Rostros","Sonrisas","Ojos","Canalizar","Cartonizar"]
    Feature_choice =  st.sidebar.selectbox("Encuentra características",task)
    
    if st.button("Procesar"):
      if Feature_choice == "Rostros":
        st.title("Rostros")
        result_faces, result_img = detect_face(our_image)
        st.image(result_img, use_column_width=True)
        st.success("Rostros encontrados: {}".format(len(result_faces)))
      
      if Feature_choice == "Sonrisas":
        st.title("Sonrisas")
        result_smiles, result_img = detect_smile(our_image)
        st.image(result_img, use_column_width=True)
        st.success("Sonrisas encontrados: {}".format(len(result_smiles)))
      
      if Feature_choice == "Ojos":
        st.title("Ojos")
        result_eyes, result_img = detect_eye(our_image)
        st.image(result_img, use_column_width=True)
        st.success("Rostros encontrados: {}".format(len(result_eyes)))
      
      if Feature_choice == "Canalizar":
        st.title("Canalizar")
        result_img = cannize_image(our_image)
        st.image(result_img, use_column_width=True)
      
      if Feature_choice == "Cartonizar":
        st.title("Cartonizar")
        result_img = cartonize_image(our_image)
        st.image(result_img, use_column_width=True)
   
    enhance = ["Original","Escala de grises","Contraste","Brillantez","Difuminado"]
    enhance_type = st.sidebar.radio("Tipo de mejora",enhance)
    
    if enhance_type == 'Escala de grises':
      st.title("Gray-scale")
      new_img = np.array(our_image.convert('RGB'))
      img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
      st.image(img, caption="Image subida por el usuario", use_column_width=True )

    if enhance_type == 'Contraste':
      st.title("Contrast")
      c_rate = option_contrast(st)
      enhancer = ImageEnhance.Contrast(our_image)
      img_output = enhancer.enhance(c_rate)
      st.image(img_output,use_column_width=True)
      
    if enhance_type == 'Brillantez':
      st.title("Brightness")
      c_rate = option_contrast(st)
      enhancer = ImageEnhance.Brightness(our_image)
      img_output = enhancer.enhance(c_rate)
      st.image(img_output,use_column_width=True) 
      
    if enhance_type == 'Difuminado':
      st.title("Blurring")
      c_rate = option_contrast(st)
      new_img = np.array(our_image.convert('RGB'))
      img = cv2.GaussianBlur(new_img,(11,11),c_rate)
      st.image(img, caption="Image subida por el usuario", use_column_width=True )         
          
  elif choice == "Acerca":
    st.subheader("About")


if __name__ == '__main__':
    main()
    