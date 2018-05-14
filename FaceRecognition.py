# -*- coding: utf-8 -*-
import dlib
import cv2
import numpy as np
import glob


root = '/home/tas/anaconda2/lib/python2.7/site-packages/face_recognition_models/models/'
pose_predictor_68_point = dlib.shape_predictor(root+'shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1(root+'dlib_face_recognition_resnet_model_v1.dat')



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
def draw_landmark(img, landmarks):
    cur_img = np.copy(img)
    for mark in landmarks:
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


def find_face_by_img(img):
    img_search = np.copy(img)
    img_search = cv2.cvtColor(img_search, cv2.COLOR_BGR2RGB)
    faces, landmark = find_face(img_search)
    draw_img = draw_rectangle(img, faces)
    #draw_img = draw_landmark(draw_img, landmark)
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


def GetTheNearFace(curLandMark, tolerance=0.6):
    names = []
    for face in allFace:
        distant = face_distance(face[1], curLandMark)
        if distant <= tolerance:
            names.append((face[0], distant))
    names = sorted(names, cmp=lambda x, y: cmp(x[1], y[1]))
    return names


def find_face_by_video():
    cap = cv2.VideoCapture(0)
    n = 0
    while cap.isOpened():
        n += 1
        ret, img = cap.read()
        if n % 1 == 0:
            draw_img = find_face_by_img(img)
        else:
            draw_img = img
        cv2.imshow('face', draw_img)
        k = cv2.waitKey(10)
        if k == 27:
            break


if __name__ == '__main__':
    if True:
        img = cv2.imread('/home/tas/桌面/22.jpg')
        draw_img = find_face_by_img(img)
        cv2.imshow('face', draw_img)
        cv2.waitKey(0)
    else:
        find_face_by_video()