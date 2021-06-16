import face_recognition as fr
#from PIL import Image, ImageDrow
from PIL import Image,ImageDraw

def face_rec():
	img = fr.load_image_file('PJ.jpg')
	face_loc = fr.face_locations(img)
	
	imgs = fr.load_image_file('bbt.jpg')
	faces_loc = fr.face_locations(imgs)
	
	print(f"Found {len(faces_loc)} face(s) in file")
	
	pil_img = Image.fromarray(imgs)
	drawl = ImageDraw.Draw(pil_img)
	
	for(top, right, button, left) in faces_loc:
		drawl.rectangle(((left, top),(right, button)),outline=(255,255,0),width=4)
	#print(faces_loc)
	del drawl
	pil_img.save('./new.jpg')

def extracr_face(img_path):
	count = 0
	faces = fr.load_image_file(img_path)
	print(faces)
	faces_locations = fr.face_locations(faces)
	for face_locations in faces_locations:
		top,right,bottom,left = face_locations
		face_img = faces[top:bottom, left:right]
		pil_img = Image.fromarray(face_img)
		pil_img.save(f"./new_{count}.jpg")
		count+=1
	return(count)	

def compare_fases(img_path, img2_path):
	img1 = fr.load_image_file(img_path)
	img2 = fr.load_image_file(img2_path)
	img1_enc = fr.face_encodings(img1)
	img2_enc = fr.face_encodings(img2)
	print(img1_enc)
	result = fr.compare_faces(img1_enc[1],img2_enc)
	print(result)
#extracr_face('bbt.jpg')	
compare_fases('bbt.jpg','new_1.jpg')