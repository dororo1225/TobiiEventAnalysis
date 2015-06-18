# coding:utf-8
__author__ = 'yamamoto'
# env: Python2.7

import cv2
import numpy as np
import pandas as pd
import os
import math
import time

################################################
#  初期入力
################################################

# # 出力の指定
OutputPic = 1    # Whether Output Movie (1:Output / 0:Don't Output)
OutputCSV = 1  # Whether OutputCSV (1:Output / 0:Don't Output)


# # 入力の指定(Video Input)
video = "hogehoge.mp4"


# # 視線データの指定(関数preprocessの引数)
ParName = 'HogeHoge'  # like 'YamamotoHiroki'
RecDate = '2015-04-23'
RecName = 'Recording011'


# # イベント名
EventType = 'EC'


# # 画像のresizeの倍率
prop_resize = 1 * 1.0 / 2


# # イベント前後のtime windowの幅(FrameWindow=2: イベントの前後2フレーム)
FrameWindow = 0


# # 画像に加える回転のrotate window
interval = 10  # 回転角度の間隔
rotateNum = 5  # 回転の回数


# # 視線フィルタの引数
flagGF = 1 # 視線フィルタの有無
region = 1.5  # FalseAlarmを減らす、Gazeフィルタのパラメーター


# # Choose Face Detector
# cascade_path = "C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml"
cascade_path = "C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml"
# cascade_path = "C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml"
# cascade_path = "C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt_tree.xml"


################################################
#  処理関数
################################################

# # 関数preprocess
def preprocess(x, PName, Date, RName):
    x = x[(x['Participant name'] == PName) & (x['Recording date'] == Date) & (x['Recording name'].str.startswith(RName))]
    return x


# 関数rotateGaze
def rotateGaze(gaze_x, gaze_y, c_x, c_y, theta):
    # gaze_x  [int]: 固視点のX座標
    # gaze_y  [int]: 固視点のY座標
    # c_x     [int]: 回転中心のX座標
    # c_y     [int]: 回転中心のY座標
    # theta   [int]: 回転角度(degree)
    theta = - theta
    Cos = math.cos(math.radians(theta))
    Sin = math.sin(math.radians(theta))
    M = np.array([[Cos, - Sin, c_x - c_x * Cos + c_y * Sin],
                  [Sin, Cos, c_y - c_x * Sin - c_y * Cos],
                  [0, 0, 1]])
    gaze_vec = np.array([gaze_x, gaze_y, 1])
    gaze_rot = M .dot(gaze_vec)  # 行列の積算
    return [gaze_rot[0], gaze_rot[1]]


# 関数annotateFaceID
def annotateFaceID(x_an, y_an, wid, hei, image, id):
    # x_an    [int?]: アノテーション(枠)の左上のX座標
    # y_an    [int?]: アノテーション(枠)の左上のY座標
    # wid     [int?]: アノテーション(枠)の幅
    # hei     [int?]: アノテーション(枠)の高さ
    # image [imaage]: 画像
    # id       [int]: FaceID
    id += 1
    cv2.rectangle(image, (x_an, y_an), (x_an + wid, y_an + hei), (196, 191, 0), 2)
    msg = 'F-ID:' + str(id)
    cv2.putText(image, msg, (x_an, y_an - 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (196, 191, 0), 1)
    msg = 'Detected'
    cv2.putText(image, msg, (0, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (196, 191, 0), 1)
    return [image, id]


# 関数noFace
def noFace(image):
    msg = 'No Face'
    cv2.putText(image, msg, (0, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (109, 118, 248), 1)



################################################
#  メイン処理
################################################

# # ビデオの読み込み・プロパティの取得
cap = cv2.VideoCapture(video) # Read Video File
width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
FrameRate = float(cap.get(cv2.cv.CV_CAP_PROP_FPS))
FrameNum_Total = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, FrameNum_Total)
MaxTimeSTP = int(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
print 'FrameRate', FrameRate
print 'MaxFrameNumber', FrameNum_Total
print 'MaxTimeStamp', MaxTimeSTP
print


# # 視線データの読み込み
RecName = RecName if 'RecName' in locals() else 'Recording'
reader = pd.read_csv('Data Export.tsv', delimiter='\t', chunksize=1000)
df_gaze = pd.concat((preprocess(r, ParName, RecDate, RecName) for r in reader), ignore_index=True)
df_event = df_gaze[df_gaze['Event'] == EventType]  # select Event
print 'df_gaze'
print df_gaze.shape
print df_gaze.head(1)
print


# #  Eye ContactのTimwWindowとRotate Windowも含めたDataFrameをつくる(例外処理をつける)
# EyeContactの前後FrameWindow(2フレーム)を含めたDataFrameを作成
EventTimeSTP = df_event[['Recording timestamp', 'Fixation point X', 'Fixation point Y', 'Recording media width', 'Recording media height']]
EventTimeSTP.index = range(0, EventTimeSTP.shape[0])
ID_frame = pd.DataFrame({'EventID': range(0, EventTimeSTP.shape[0]),
                         'FrameID': np.zeros(EventTimeSTP.shape[0])})
origin = pd.concat([EventTimeSTP, ID_frame], axis=1)
before_upd = pd.concat([EventTimeSTP, ID_frame], axis=1)
after_upd = pd.concat([EventTimeSTP, ID_frame], axis=1)
for var in range(1, FrameWindow + 1):
    before_upd[['Recording timestamp']] -= 1 / FrameRate * 1000
    before_upd[['FrameID']] -= 1
    after_upd[['Recording timestamp']] += 1 / FrameRate * 1000
    after_upd[['FrameID']] += 1
    origin = pd.concat([origin, before_upd, after_upd])
origin = origin[(origin['Recording timestamp'] >= 0) & (origin['Recording timestamp'] < MaxTimeSTP)] # 例外を除く
origin = origin.sort("Recording timestamp")
origin.index = range(0, origin.shape[0])
# 各フレームに回転を加えた画像を含めたDatFrameを作成
ID_rotate = pd.DataFrame({'RotateID': np.zeros(origin.shape[0])})
before_rot = pd.concat([origin, ID_rotate], axis=1)
after_rot = pd.concat([origin, ID_rotate], axis=1)
origin = pd.concat([origin, ID_rotate], axis=1)
for var in range(1, rotateNum + 1):
    before_rot[['RotateID']] -= interval
    after_rot[['RotateID']] += interval
    origin = pd.concat([origin, before_rot, after_rot])
origin = origin.sort(['Recording timestamp', 'FrameID', 'RotateID'])
origin.index = range(0, origin.shape[0])
origin = origin[['Recording timestamp',
                 'EventID',
                 'FrameID',
                 'RotateID',
                 'Fixation point X',
                 'Fixation point Y',
                 'Recording media width',
                 'Recording media height']]

# # Choose Face Detector
cascade = cv2.CascadeClassifier(cascade_path)
# 顔検出画像の保存ディレクトリの作成
if not os.path.isdir('output'):
    os.mkdir('output')


########################
#  # Face Detection
########################

out_list = []  # Logの準備
for idx in origin.index:
    TimeSTP_set = origin.ix[idx, 'Recording timestamp']
    EventID = origin.ix[idx, 'EventID']
    FrameID = int(origin.ix[idx, 'FrameID'])
    RotateID = int(origin.ix[idx, 'RotateID'])
    Gaze_x = origin.ix[idx, 'Fixation point X']
    Gaze_y = origin.ix[idx, 'Fixation point Y']

    # 画像の切り出し
    cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, TimeSTP_set)
    FrameNum = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    # cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, Frame_Num - 1)
    # FrameNum = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    MovieTimeStamp = int(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
    ret, im = cap.read()

    if ret:
        # # グレースケール化・輝度を正規化
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # gray scale
        # im = cv2.equalizeHist(im)  # 輝度を正規化

        # 画像の回転
        height, width, cha = im.shape
        M = cv2.getRotationMatrix2D((width / 2, height / 2), RotateID, 1)
        im = cv2.warpAffine(im, M, (width, height))

        # 視線の回転
        Gaze_x, Gaze_y = rotateGaze(Gaze_x, Gaze_y, width / 2, height / 2, RotateID)

        # 画像のリサイズ
        im_resize = cv2.resize(im, (int(im.shape[1] * prop_resize), int(im.shape[0] * prop_resize)))

        f_id = 0  # FaceID
        face_resize = cascade.detectMultiScale(im_resize, 1.1, 2)  # face detection

        # アノテーションを画像に付加して描画
        if OutputPic == 1:
            # 画像の左上にEventID, FrameID, RotateIDを記載する
            cv2.rectangle(im_resize, (0, 0), (180, 50), (50, 50, 50), -1)
            title = 'E-ID:' + str(EventID) + ',' + str(FrameID) + ',' + str(RotateID)

            # 画像上に視点データを描画する
            cv2.putText(im_resize, title, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)
            cv2.circle(im_resize, (int(Gaze_x * prop_resize), int(Gaze_y * prop_resize)), 8, (109, 118, 248), 1)

            # 枠の描画
            if len(face_resize) == 0:  # 顔が検出されなかった場合　
                noFace(im_resize)

            else:  # 顔が検出された場合
                for (x, y, w, h) in face_resize:
                    if flagGF == 0:  # 視線フィルタなし
                        im, f_id = annotateFaceID(x, y, w, h, im_resize, f_id)
                    else:  # 視線フィルタあり
                        if (Gaze_x * prop_resize - (x + w / 2)) ** 2 + (Gaze_y * prop_resize - (y + h / 2)) ** 2 \
                                <= (region * h / 2) ** 2:
                            im, f_id = annotateFaceID(x, y, w, h, im_resize, f_id)
                # 視線フィルタによってHitがなくなった場合
                if f_id == 0:
                    noFace(im_resize)

            # 画像を描画
            cv2.imshow('Video Stream', im_resize)
            # キー入力待機
            # cv2.waitKey(0)

            # 画像の書き出し
            if FrameID < 0:
                sign_f = 'm'
            else:
                sign_f = ''
            if RotateID < 0:
                sign_r = 'm'
            else:
                sign_r = ''
            file_path = 'output/' + EventType + str(idx) + '_' + 'Event' + \
                        str(EventID) + '.' + str(abs(FrameID)) + sign_f + '.' + str(abs(RotateID)) + sign_r + '.jpg'
            cv2.imwrite(file_path, im_resize)

        # CSVの書き出し
        f_id = 0  # FaceID
        if OutputCSV == 1:
            # 枠の描画
            if len(face_resize) == 0:  # 顔が検出されなかった場合　
                out_list.append([video, EventID, FrameID, RotateID, TimeSTP_set, FrameNum, MovieTimeStamp, Gaze_x, Gaze_y, None, None, None, None, None])
            else:  # 顔が検出された場合
                for (x, y, w, h) in face_resize:
                    if flagGF == 0:  # 視線フィルタなし
                        f_id += 1
                        out_list.append([video, EventID, FrameID, RotateID, TimeSTP_set, FrameNum, MovieTimeStamp, Gaze_x, Gaze_y, f_id, x, y, w, h])
                    else:  # 視線フィルタあり
                        if (Gaze_x * prop_resize - (x + w / 2)) ** 2 + (Gaze_y * prop_resize - (y + h / 2)) ** 2 \
                                <= (region * h / 2) ** 2:
                            f_id += 1
                            out_list.append([video, EventID, FrameID, RotateID, TimeSTP_set, FrameNum, MovieTimeStamp, Gaze_x, Gaze_y, f_id, x, y, w, h])
                if f_id == 0:
                    out_list.append([video, EventID, FrameID, RotateID, TimeSTP_set, FrameNum, MovieTimeStamp, Gaze_x, Gaze_y, None, None, None, None, None])

    if cv2.waitKey(10) > 0:
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()  # キャプチャー解放
cv2.destroyAllWindows()  # ウィンドウ破棄


if OutputCSV == 1:
    df_face = pd.DataFrame(out_list)
    print 'df_face'
    print df_face.head()
    print
    df_face.columns = ['VideoName',
                       'EventID',
                       'FrameID',
                       'RotateID',
                       'GazeTimeStamp',
                       'FrameNum',
                       'MovieTimeStamp',
                       'Gaze_x',
                       'Gaze_y',
                       'FaceID',
                       'pos_x',
                       'pos_y',
                       'Width',
                       'Height']
    df_face.to_csv('FacePosition.csv', index=False, na_rep='NA')