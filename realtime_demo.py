"""
Face detection
"""
import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import time
import hashlib  
import random  
import string
import requests  
import base64  
import requests
from urllib.parse import urlencode
import json #用于post后得到的字符串到字典的转换

app_id = '1106649312' 
app_key = 'TwsQQv5G5c9E6FsH'


class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.5, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x-64, y-64 ), (x+96 , y-32), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, label, (x-48, y-48 ), font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        
        # infinite loop, break by key ESC
        while True:
            if not video_capture.isOpened():
                sleep(5)
            
            ret, frame = video_capture.read()
            if int(time.time()-starttime)%8==0:
                nparry_encode = cv2.imencode('.jpg', frame)[1]
                data_encode = np.array(nparry_encode)
                img = base64.b64encode(data_encode)    #得到API可以识别的字符串

                params = get_params(img)    #获取鉴权签名并获取请求参数

                url = "https://api.ai.qq.com/fcgi-bin/face/face_detectface"  # 人脸分析
                #检测给定图片（Image）中的所有人脸（Face）的位置和相应的面部属性。位置包括（x, y, w, h），面部属性包括性别（gender）, 年龄（age）, 表情（expression）, 魅力（beauty）, 眼镜（glass）和姿态（pitch，roll，yaw）   
                res = requests.post(url,params).json()
                for obj in res['data']['face_list']:
                #print(obj)
                    global xx
                    global yy
                    global ww
                    global hh
                    
                    xx=obj['x']
                    yy=obj['y']
                    ww=obj['width']
                    hh=obj['height']
                    cv2.rectangle(frame,(xx,yy),(xx+ww,yy+hh),(255,255,255),2)
                    #delt=hh//5
                    global gender
                    global age
                    global smile
                    global beauty
                    global glass
                    
                    gender=str(obj['gender'])
                    age=str(obj['age'])
                    smile = str(obj['expression'])
                    beauty=str(obj['beauty'])
                    glass=str(obj['glass'])
            cv2.putText(frame,'tencent API', (xx+ww+10, yy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,cv2.LINE_8, 0)
            cv2.putText(frame,'gen:  '+ gender, (xx+ww+10, yy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,cv2.LINE_8, 0)
            cv2.putText(frame,'age   : '+ age, (xx+ww+10, yy+10+hh//5*1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,cv2.LINE_8, 0)
            cv2.putText(frame,'smile : '+ smile, (xx+ww+10, yy+10+hh//5*2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,cv2.LINE_8, 0)
            cv2.putText(frame,'beauty: '+ beauty, (xx+ww+10, yy+10+hh//5*3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,cv2.LINE_8, 0)
            cv2.putText(frame,'glass : '+ glass, (xx+ww+10, yy+10+hh//5*4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,cv2.LINE_8, 0)


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=10,minSize=(self.face_size, self.face_size))
            
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_imgs[i,:,:,:] = face_img
            global label
            global starttime
            

            if len(face_imgs) > 0 :
                   # predict ages and genders of the detected faces
                if int(time.time()-starttime)%10==0:
                   results = self.model.predict(face_imgs)
                   predicted_genders = results[0]
                   ages = np.arange(0, 101).reshape(101, 1)
                   predicted_ages = results[1].dot(ages).flatten()
                   
                # draw results
            for i, face in enumerate(faces):	
                if int(time.time()-starttime)%10==0: 			
                    label = "age={}, gen={}".format(int(predicted_ages[i]), "F" if predicted_genders[i][0] > 0.5 else "M")
                    
                self.draw_label(frame, (face[0], face[1]), label)
                
            cv2.imshow('Keras Faces', frame)
              
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args
	
def get_params(img):                         #鉴权计算并返回请求参数
    #请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效
    time_stamp=str(int(time.time())) 
    #请求随机字符串，用于保证签名不可预测,16代表16位
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    params = {'app_id':app_id,                #请求包，需要根据不同的任务修改，基本相同
              'image':img,                    #文字类的任务可能是‘text’，由主函数传递进来
              'mode':'0' ,                    #身份证件类可能是'card_type'
              'time_stamp':time_stamp,        #时间戳，都一样
              'nonce_str':nonce_str,          #随机字符串，都一样
              #'sign':''                      #签名不参与鉴权计算，只是列出来示意
             }

    sort_dict= sorted(params.items(), key=lambda item:item[0], reverse = False)  #字典排序
    sort_dict.append(('app_key',app_key))   #尾部添加appkey
    rawtext= urlencode(sort_dict).encode()  #urlencod编码
    sha = hashlib.md5()    
    sha.update(rawtext)
    md5text= sha.hexdigest().upper()        #MD5加密计算
    params['sign']=md5text                  #将签名赋值到sign
    return  params                          #返回请求包


def main():
    args = get_args()
    depth = args.depth
    width = args.width
    global starttime
    global label
    global gender
    global age
    global smile
    global beauty
    global glass
    global xx
    global yy
    global ww
    global hh
    label=""
    starttime=time.time()
    gender='0'
    age='0'
    smile='0'
    beauty='0'
    glass='0'
    xx=0
    yy=0
    ww=0
    hh=0
    
    face = FaceCV(depth=depth, width=width)

    face.detect_face()

if __name__ == "__main__":
    main()
