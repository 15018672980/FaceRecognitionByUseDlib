# -*- coding: utf-8 -*-
import dlib
import cv2
import numpy as np
import glob
from addEye import *


root = '/home/tas/anaconda2/lib/python2.7/site-packages/face_recognition_models/models/'
pose_predictor_68_point = dlib.shape_predictor(root+'shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1(root+'dlib_face_recognition_resnet_model_v1.dat')
left_eye = cv2.imread("/home/tas/桌面/lefteye00.png")
right_eye = cv2.imread("/home/tas/桌面/righteye00.png")
lcenter = None
rcenter = None

def face_detect(img):
    dets = face_detector(img, 1)
    return [d for i, d in enumerate(dets)]


def face_landmark(img, faces):
    faceLandmarks = []
    for face in faces:
        shape = pose_predictor_68_point(img, face)
        faceLandmarks.append(shape)
    return faceLandmarks


def find_face(img):
    cur_img = np.copy(img)
    faces = face_detect(cur_img)
    landmarks = face_landmark(cur_img, faces)
    return faces, landmarks


# 绘制关键点
def draw_landmark(img, landmarks,drawMark= False, drawEye = True):
    cur_img = np.copy(img)
    for mark in landmarks:
        if drawEye:
            lp1 = mark.part(36)
            lp2 = mark.part(39)
            rp1 = mark.part(42)
            rp2 = mark.part(45)
            lcenter,cur_img = put_logo_in_img(cur_img, left_eye,(lp1.x,lp1.y),(lp2.x,lp2.y),0,-5)
            rcenter,cur_img = put_logo_in_img(cur_img, right_eye, (rp1.x, rp1.y), (rp2.x, rp2.y),0,5)
        if drawMark:
            for i in range(mark.num_parts):
                point = mark.part(i)
                cv2.circle(cur_img, (point.x, point.y), 3, (0, 0, 255))
    return cur_img


# 绘制矩阵
def draw_rectangle(img, rectangles):
    cur_img = np.copy(img)
    for r in rectangles:
        cv2.rectangle(cur_img, (r.left(), r.top()), (r.right(), r.bottom()), (0, 0, 255))
    return cur_img


def put_name(img, draw_img, landmarks):
    for landmark in landmarks:
        curCoding = face_encodings(img, [landmark])
        persons = GetTheNearFace(curCoding)
        if len(persons)>0:
            person = persons[0]
            name = person[0]
            cv2.putText(draw_img, name, (landmark.rect.left(), landmark.rect.top()),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)
    return draw_img


templandmarks = []
tempfaces = []


def find_face_by_img(img, if_find_face=True):
    global templandmarks, tempfaces
    img_search = np.copy(img)
    img_search = cv2.cvtColor(img_search, cv2.COLOR_BGR2RGB)
    faces =[]
    if if_find_face:
        faces, landmark = find_face(img_search)
        templandmarks, tempfaces = faces, landmark
    if not faces:
        faces, landmark = templandmarks, tempfaces
    if not faces:
        return img
    draw_img = draw_rectangle(img, faces)
    draw_img = draw_landmark(draw_img, landmark)
    draw_img = put_name(img_search, draw_img, landmark)
    return draw_img


def face_encodings(face_image, landmarks, num_jitters=1):
    return [np.array(face_encoder.compute_face_descriptor(face_image, landmark, num_jitters)) for landmark in landmarks]


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0 or len(face_to_compare) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def GetAllFace(url):
    img_urls = glob.glob(url+'/*.jpg')
    img_landmarks = []
    for img_url in img_urls:
        img = cv2.imread(img_url)
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces, landmark = find_face(img_RGB)
        name = img_url.replace(url+'/', '').replace('.jpg', '')

        img_landmarks.append((name, face_encodings(img_RGB, landmark)[0]))
    return img_landmarks


allFace = GetAllFace('/home/tas/PycharmProjects/untitled/venv2/dlibFace/faceDB')


def GetTheNearFace(curLandMark, tolerance=0.7):
    names = []
    for face in allFace:
        distant = face_distance(face[1], curLandMark)
        if distant <= tolerance:
            names.append((face[0], distant))
    names = sorted(names, cmp=lambda x, y: cmp(x[1], y[1]))
    return names


def find_face_by_video():
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('/home/tas/下载/saveVideo.avi', fourcc, fps, (442, 240))
    cap = cv2.VideoCapture("/home/tas/下载/38186707-1-6.mp4")
    n = 0
    while cap.isOpened():
        n += 1
        ret, img = cap.read()
        if img is None:
            break
        draw_img = find_face_by_img(img, n % 2 == 0)

        cv2.imshow('face', draw_img)
        videoWriter.write(draw_img)
        k = cv2.waitKey(10)
        if k == 27:
            break


if __name__ == '__main__':
    if False:
        img = cv2.imread('/home/tas/桌面/下载.jpg')
        draw_img = find_face_by_img(img)
        cv2.imshow('face', draw_img)
        cv2.imwrite('/home/tas/桌面/result2.jpg', draw_img)
        cv2.waitKey(0)
    else:
        find_face_by_video()