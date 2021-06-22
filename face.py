import face_recognition as fr
import os
from PIL import Image,ImageDraw
import sys
import time
import pickle
import cv2 as cv

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
	
def object_person_in_video():
		
		data = pickle.loads(open("SANA_encodings",'rb').read())
		#print(data)
		video_file = cv.VideoCapture("./vidos/1a.mp4")
		while True:
			ret, image = video_file.read()
			loc = fr.face_locations(image)# координаты морды верх-права низ-лево
			#loc = fr.face_locations(image, model = "cnn")# координаты морды верх-права низ-лево
			encode = fr.face_encodings(image, loc)
			
			for face_enc, face_loc in zip(encode, loc):
				result = fr.compare_faces(data['encoding'], face_enc)
				match = None
				
				if True in result:
					match = data['name']
					#print(f'Sovpadenie na vidio -{match}- ')
					left_top = (face_loc[3], face_loc[0])
					right_bottom = (face_loc[1], face_loc[2])
					colorR = [0,0,255]
					cv.rectangle(image, left_top,right_bottom,colorR)
					cv.putText(image, 'Sana maniak_fanatic', (face_loc[3]+10, face_loc[2]+40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				else:
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
			
def train_nodel_by_img():
	if not os.path.isdir('dataset'):
		print("[ERROR] there is dir")
		sys.exit()

	known_encodings = []
	images =  os.listdir('dataset')
	for(i, image) in enumerate(images):
	
		print(f"[+]{i}\n")
		img1 = fr.load_image_file(f"dataset/{image}")# LOAD
		img1_enc = fr.face_encodings(img1)[0]# pervi element ubiraet array CODE
		#print(img1_enc)
		if len(known_encodings)==0:
			print("pust")
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