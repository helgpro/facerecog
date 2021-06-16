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
		data = pickle.loads(open("./SANA_encodings",'wb').read())
		print(data)
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