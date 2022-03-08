import cv2
import xml.etree.ElementTree as ET 
import numpy as np
from PIL import Image,ImageChops
from glob import glob
from tqdm import tqdm 

def read_xml(xml_path):
	'''读取标注区域坐标'''
	info=ET.parse(xml_path).getroot()
	location=[]
	for obj in info.iter("object"):
		box=obj.find("bndbox")
		xmin=int(float(box.find("xmin").text))
		ymin=int(float(box.find("ymin").text))
		xmax=int(float(box.find("xmax").text))
		ymax=int(float(box.find("ymax").text))
		location.append((xmin,ymin,xmax,ymax))
	return location

def crop_defect(img_path,location):
	'''将缺陷区域切割下来'''
	img=cv2.imread(img_path)
	defect_regions=[]	
	for locate in location:
		xmin,ymin,xmax,ymax=locate
		crop_img=img[ymin:ymax,xmin:xmax,:]
		defect_regions.append(crop_img)
	return defect_regions,location

def generate_new_img(ori_img_path,defect_regions,defect_locaton,save_path):
	'''缺陷区域与原图拼接'''
	ori_img=cv2.imread(ori_img_path)
	filename=ori_img_path.split("/")[-1]
	save_img=ori_img.copy()
	for index,defect in enumerate(defect_regions):
		location=defect_locaton[index]
		xmin,ymin,xmax,ymax=location
		ori_img[ymin:ymax,xmin:xmax,:]=defect 
	# save_img=save_img[302:695,150:504,:]
	# ori_img=ori_img[302:695,150:504,:]

	cv2.imwrite(save_path+"/{}".format(filename),save_img)	
	cv2.imwrite(save_path+"/{}".format(filename.replace("temp","test")),ori_img)	
	

def check_img(new_img,ori_img,diff_save_location):
	'''检查前后图片质量是否改变'''
	image_one=Image.open(new_img)
	image_two=Image.open(ori_img)
	try:
		diff=ImageChops.difference(image_one,image_two)
		if diff.getbbox() is None:
			print("same..........")
		else: 
			diff.save(diff_save_location)
	except ValueError as e:
		print(e)
	

# origin_path="./data/VOCdevkit/0000_temp.jpg"
# defect_path="./data/VOCdevkit/0000_test.jpg"
# defect_xml="./data/VOCdevkit/0000_test.xml"

# locations=read_xml(defect_xml)
# defect_regions,defect_locaton=crop_defect(defect_path,locations)
# generate_new_img(origin_path,defect_regions,defect_locaton)
# check_img("./data/VOCdevkit/def_0000_temp.jpg","./data/VOCdevkit/ori_0000_temp.jpg","./")

root_path="../data\VOCdevkit\data4.1/"
img_paths=glob(root_path+"/JPEGImages/*test.jpg")
save_path="./data/VOCdevkit/newdata4.1/"

for img_path in tqdm(img_paths):
	defect_xml=img_path.replace("JPEGImages","Annotations").replace("jpg","xml")
	locations=read_xml(defect_xml)
	defect_regions,defect_locaton=crop_defect(img_path,locations)
	origin_path=img_path.replace("test","temp")
	generate_new_img(origin_path,defect_regions,defect_locaton,save_path)