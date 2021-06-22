# Распознование лиц с потока видео facerecognition

>  установка на винду 

      pip install cmake
      pip install opencv-python
> Скачать visual studio component поставить галки cmake ++c Build 
>В папке face-recognition запустить фаил pip install dlib-19.19.0-cp37-cp37m-win_amd64.whl

      pip install face-recognition
      pip install dlib
      
### фаил разбит на функции 

импортируем моуль создаем объект as fr
def face_rec():  
  img = fr.load_image_file('PJ.jpg') # загружаем файл
	face_loc = fr.face_locations(img) # получить координаты морды
	
	imgs = fr.load_image_file('bbt.jpg')
	faces_loc = fr.face_locations(imgs)
	
	print(f"Found {len(faces_loc)} face(s) in file")
	# нарисовать квадрат на морде в картинке 
	pil_img = Image.fromarray(imgs)
	drawl = ImageDraw.Draw(pil_img)
	
	for(top, right, button, left) in faces_loc:
		drawl.rectangle(((left, top),(right, button)),outline=(255,255,0),width=4)
	#print(faces_loc)
	del drawl
	pil_img.save('./new.jpg')
> Функция получить метки на морду

def compare_fases(img_path, img2_path):
	img1 = fr.load_image_file(img_path)
	img2 = fr.load_image_file(img2_path)
	img1_enc = fr.face_encodings(img1)  # Получить метки
	img2_enc = fr.face_encodings(img2)
	print(img1_enc)
	result = fr.compare_faces(img1_enc[1],img2_enc) # Сравнить метки
	print(result)
  
  ## Работа с видосом
> имортируем модули  
import face_recognition as fr
import os
from PIL import Image,ImageDraw
import sys
import time
import pickle
import cv2 as cv

> Получаем видео разбиваем на фрэймы

def take_screenshot_from_vidio():
	cap = cv.VideoCapture("./vidos/1.mp4")
	if not cap.isOpened():
		print("Cannot open camera")
		exit()
		#ap = cv2.VidioCapture('./vidos/1.mp4')
	while True:
			ret, frame = cap.read()
			if ret:
				cv.imshow('frame',frame)
				cv.waitKey(1)
			else:
				break
				
	cap.release()
	cv.destroyAllWindows()

>Функция загрузки меток и сравниванее с фрэймом на совпадение 	
def object_person_in_video():
		
		data = pickle.loads(open("SANA_encodings",'rb').read()) # Загрузка меток в  train_nodel_by_img() набираем данные
		#print(data)
		video_file = cv.VideoCapture("./vidos/1a.mp4")
		while True:
			ret, image = video_file.read()
			loc = fr.face_locations(image)# координаты морды верх-права низ-лево
			#loc = fr.face_locations(image, model = "cnn")# координаты морды верх-права низ-лево
			encode = fr.face_encodings(image, loc) # метки
			
			for face_enc, face_loc in zip(encode, loc):
				result = fr.compare_faces(data['encoding'], face_enc) #  Сравнение
				match = None # переменная флаг потом задействовать сократит код
				
				if True in result: # Совпало по меткам
					match = data['name'] # на несколько персонажей в плане набора данных из файла
					#print(f'Sovpadenie na vidio -{match}- ')
					left_top = (face_loc[3], face_loc[0])
					right_bottom = (face_loc[1], face_loc[2])
					colorR = [0,0,255]
					cv.rectangle(image, left_top,right_bottom,colorR)# рамка
					cv.putText(image, 'Sana maniak_fanatic', (face_loc[3]+10, face_loc[2]+40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)# текст
				else: # не совпало 
					print("tip ne opoznan")
					left_top = (face_loc[3], face_loc[0])
					right_bottom = (face_loc[1], face_loc[2])
					color = [0,255,0]
					cv.rectangle(image, left_top,right_bottom,color)
					cv.putText(image, 'poc neopoznan', (face_loc[3], face_loc[2]+30), cv.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
				cv.imshow("nazvanie",image)	
				k = cv.waitKey(1)
				if k== ord("q"):
					print("q press clossed app")
					break
			
def train_nodel_by_img(): # Функция набирает метки с персонажей по фото
	if not os.path.isdir('dataset'): # Проверить дир
		print("[ERROR] there is dir")
		sys.exit()

	known_encodings = []
	images =  os.listdir('dataset') # Созд дир
	for(i, image) in enumerate(images):
	
		print(f"[+]{i}\n")
		img1 = fr.load_image_file(f"dataset/{image}")# LOAD
		img1_enc = fr.face_encodings(img1)[0]# pervi element ubiraet array CODE
		#print(img1_enc)
		if len(known_encodings)==0:
			print("pust списчик
			known_encodings.append(img1_enc)
		else:
			for item in range(0, len(known_encodings)):
				#print('imegeinc[]',img1_enc)
				result = fr.compare_faces([img1_enc],known_encodings[item])#[] добавляет array
				
				print(result)
				
				if result[0]:
					known_encodings.append(img1_enc)
					break
				else:
						print('not sovpadeni')
	data = {
		"name":"Sanya",
		"encoding":known_encodings
			}					
	with open("SANA_encodings",'wb')as file:
		file.write(pickle.dumps(data))
		
			
	
	#print('encoding TRUE')				
	#print(known_encodings)				
	#print(known_encodings)	

#train_nodel_by_img()
#take_screenshot_from_vidio()
object_person_in_video()
