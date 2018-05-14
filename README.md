# FaceRecognitionByUseDlib
使用Dlib进行人脸识别，可以识别出人脸是图库里的谁。和识别人脸68个关键点

# 首先需要安装Dlib 
- ubuntu 可以直接pip install Dlib
- 安装完后，可以在dlib下的文件夹找到对应的模型文件，修改路径就好，或者在Dlib官网直接下载：
  - 关键点检测：shape_predictor_68_face_landmarks.dat
  - 人脸特征表示：dlib_face_recognition_resnet_model_v1.dat
- 注意！代码中的路径都是我电脑本地的，请修改为合适的路径


# 人脸识别效果

图片库都放在faceDB文件下，用jpg格式，用图片上的人的名字命名图片即可，由于opencv不支持中文显示，只能用英文命名。


![Alt text](https://raw.githubusercontent.com/15018672980/FaceRecognitionByUseDlib/master/result.jpg)

# 68关键点效果

![Alt text](https://raw.githubusercontent.com/15018672980/FaceRecognitionByUseDlib/master/result2.jpg)
