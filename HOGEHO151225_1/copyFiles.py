# coding:utf-8
__author__ = 'yamamoto'
# env: Python2.7

import pandas as pd
import os
import shutil
import cv2

################################################
#  メイン処理
################################################

files = os.listdir(os.getcwd())
file_name = [file for file in files if file.startswith('Annot_')]  # 内包表記

print file_name

if not os.path.isdir('AnnotationAssistant\\FaceClipper\\static\\images'):
    os.mkdir('AnnotationAssistant\\FaceClipper\\static\\images')
else:
    file_name = []
    print '------------------------------------'
    print 'Check inside folder "FaceClipper/static"'
    print '------------------------------------'

if len(file_name[0]) != 0:
    annot = pd.read_csv(file_name[0])

    for idx in annot.index:
        src = annot.ix[idx, 'Picture']
        src = src.replace('/', '\\')

        # img = cv2.imread(src)
        # cv2.imshow("Show Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        shutil.copy(src, 'AnnotationAssistant\\FaceClipper\\static\\images')