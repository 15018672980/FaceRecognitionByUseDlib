# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math


def overlay_image(image, logo, left, top):
    image[left:left + logo.shape[0], top:top + logo.shape[1]] = logo


# 图像旋转: src为原图像， dst为新图像, angle为旋转角度
def image_rotate(img, angle):
    # 得到图像大小
    width = img.shape[1]
    height = img.shape[0]
    # 计算图像中心点
    center_x = width / 2.0
    center_y = height / 2.0

    # 获得旋转变换矩阵
    scale = 1.0
    trans_mat = cv2.getRotationMatrix2D((center_x, center_y), -angle, scale)

    # 计算新图像大小
    angle1 = angle * math.pi / 180
    a = math.sin(angle1) * scale
    b = math.cos(angle1) * scale
    out_width = int(round(height * math.fabs(a) + width * math.fabs(b)))
    out_height = int(round(width * math.fabs(a) + height * math.fabs(b)))

    # 在旋转变换矩阵中加入平移量
    trans_mat[0, 2] += int(round((out_width - width) / 2))
    trans_mat[1, 2] += int(round((out_height - height) / 2))

    # 仿射变换指定背景色
    return cv2.warpAffine(img, trans_mat, (out_width, out_height), borderValue=(255, 255, 255))


# 计算点距离
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def center_point(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


# 计算偏转角度
def calculate_line_angle(p1, p2):
    xDis = p2[0] - p1[0]
    yDis = p2[1] - p1[1]
    angle = math.atan2(yDis, xDis)
    angle = angle / math.pi * 180
    return angle


# 根据p1,p2进行旋转
def turn_img(p1, p2, logo):
    img = np.copy(logo)
    # 计算宽度（两点距离）
    width = int(distance(p1, p2) * 2.5)
    # 计算高度
    height = int((width * img.shape[0] / img.shape[1]))
    img = cv2.resize(img, (width, height))
    # 计算旋转角度
    angle = calculate_line_angle(p1, p2)
    # 旋转
    return image_rotate(img, angle)


def overlay_png(image, logo, left, top):
    logoGray = np.copy(logo)
    logoGray = cv2.cvtColor(logoGray, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(logoGray, 254, 255, cv2.THRESH_BINARY)
    mask = 255 - th
    roi = image[left:left + logo.shape[0], top:top + logo.shape[1]]
    roi[mask > 0] = 0
    logo[mask == 0] = 0
    newo = cv2.add(logo, roi)
    image[left:left + logo.shape[0], top:top + logo.shape[1]] = newo
    return image


def put_logo_in_img(img, logo, p1, p2, marginx=0, marginy=0):
    newLogo = turn_img(p1, p2, logo)
    # 计算中心点
    center = center_point(p1, p2)
    top_x = int(round(center[1] - newLogo.shape[1] / 2 + marginx));
    top_y = int(round(center[0] - newLogo.shape[0] / 2 + marginy));
    try:
        new_img = overlay_png(img, newLogo, top_x, top_y)
    except:
        new_img = img
    return center, new_img


if __name__ == '__main__11':
    imgCv = cv2.imread("/home/tas/桌面/lefteye00.png");
    imgMan = cv2.imread("/home/tas/桌面/11.jpg");
    # overlay_image(imgMan, imgCv, 100, 100)
    # imgMan = image_rotate(imgMan, 20)
    # print distance((0,0),(10,10))
    # print CalculateLineAngle((0, 0), (10, 10))
    # img = turn_img((0, 0), (10, 10), imgCv)
    # img = overlay_png(imgMan, imgCv, 100, 100)
    img = put_logo_in_img(imgMan, imgCv, (180, 180), (200, 200))
    cv2.imshow("test", img)
    cv2.waitKey(0)
