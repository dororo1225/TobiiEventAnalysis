# coding:utf-8
__author__ = 'yamamoto'
# env: Python2.7

import cv2
import numpy as np
import pandas as pd
import os
import time

################################################
#  初期入力
################################################

# # 出力の指定
OutputPic = 0  # Whether Output Movie (1:Output / 0:Don't Output)
OutputCSV = 1  # Whether OutputCSV (1:Output / 0:Don't Output)

# # 入力の指定(Video Input)
videos = "inokei.mp4"

# # 視線データの指定(関数preprocessの引数)
ParName = 'InoueSatoko'
RecDate = '2015-04-23'
RecName = 'Recording011'

# # イベント名
EventType = 'EC'

# # イベント前後のtime windowの幅(FrameWindow=2: イベントの前後2フレーム)
FrameWindow = 2


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


################################################
#  メイン処理
################################################

# # ビデオの読み込み・プロパティの取得
cap = cv2.VideoCapture(videos) # Read Video File
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
reader = pd.read_csv('DataExport.tsv', delimiter='\t', chunksize=1000)
df_gaze = pd.concat((preprocess(r, ParName, RecDate, RecName) for r in reader), ignore_index=True)
df_event = df_gaze[df_gaze['Event'] == EventType]  # select Event
print 'df_gaze'
print df_gaze.shape
print df_gaze.head(1)
print


# #  Eye Contactの前後2フレームも含めたDataFrameをつくる(例外処理をつける)
# recordingタイムスタンプを取得
# タイムスタンプを設定
EventTimeSTP = df_event[['Recording timestamp', 'Fixation point X', 'Fixation point Y', 'Recording media width', 'Recording media height']]
EventTimeSTP.index = range(0, EventTimeSTP.shape[0])
IDs = pd.DataFrame({'EventID': range(0, EventTimeSTP.shape[0]),
                    'FrameID': np.zeros(EventTimeSTP.shape[0])})
origin = pd.concat([EventTimeSTP, IDs], axis=1)
before_upd = pd.concat([EventTimeSTP, IDs], axis=1)
after_upd = pd.concat([EventTimeSTP, IDs], axis=1)
for var in range(1, FrameWindow + 1):
    before_upd[['Recording timestamp']] -= 1 / FrameRate * 1000
    before_upd[['FrameID']] -= 1
    after_upd[['Recording timestamp']] += 1 / FrameRate * 1000
    after_upd[['FrameID']] += 1
    origin = pd.concat([origin, before_upd, after_upd])
origin = origin[(origin['Recording timestamp'] >= 0) & (origin['Recording timestamp'] < MaxTimeSTP)] # 例外を除く
origin = origin.sort("Recording timestamp")
origin.index = range(0, origin.shape[0])
origin = origin[['Recording timestamp',
                 'EventID',
                 'FrameID',
                 'Fixation point X',
                 'Fixation point Y',
                 'Recording media width',
                 'Recording media height']]

# # Choose Face Detector
cascade = cv2.CascadeClassifier(cascade_path)

# 顔検出画像の保存ディレクトリの作成
if OutputPic == 1 and not os.path.isdir('output'):
    os.mkdir('output')

# Logの準備
out_list = []

for idx in origin.index:
    TimeSTP_set = origin.ix[idx, 'Recording timestamp']
    EventID = origin.ix[idx, 'EventID']
    FrameID = int(origin.ix[idx, 'FrameID'])
    Gaze_x = origin.ix[idx, 'Fixation point X']
    Gaze_y = origin.ix[idx, 'Fixation point Y']
    cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, TimeSTP_set)
    ret, im = cap.read()
    FrameNum = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    MovieTimeStamp = int(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))  # MovieTimeStamp > TimeSTP_set!!!

    if ret:
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # gray scale
        # im = cv2.equalizeHist(im)  # 輝度を正規化
        region = 1.5  # FalseAlarmを減らす、Gazeフィルタのパラメーター

        # 画像を描画・書き出し
        i = 0  # FaceID
        if OutputPic == 1:
            im_half = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))  # resize video
            face_half = cascade.detectMultiScale(im_half, 1.1, 2)  # face detect
            cv2.rectangle(im_half, (0, 0), (125, 50), (50, 50, 50), -1)
            title = 'E-ID:' + str(EventID) + ',' + str(FrameID)
            cv2.putText(im_half, title, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)
            cv2.circle(im_half, (int(Gaze_x/2), int(Gaze_y/2)), 8, (109, 118, 248), 1)
            if len(face_half) == 0:  # 顔が検出されなかった場合　
                msg = 'No Face'
                cv2.putText(im_half, msg, (0, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (109, 118, 248), 1)
            else:  # 顔が検出された場合
                for (x, y, w, h) in face_half:
                    # 検出された顔内部に視線がある場合
                    if (Gaze_x / 2 - (x + w / 2)) ** 2 + (Gaze_y / 2 - (y + h / 2)) ** 2 <= (region * h / 2) ** 2:
                        i += 1
                        cv2.rectangle(im_half, (x, y), (x + w, y + h), (196, 191, 0), 2)
                        msg = 'F-ID:' + str(i)
                        cv2.putText(im_half, msg, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (196, 191, 0), 1)
                        msg = 'Detected'
                        cv2.putText(im_half, msg, (0, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (196, 191, 0), 1)
                else:
                    if i == 0:  # 検出された顔が全てFalseAlarmだった場合
                        msg = 'No Face'
                        cv2.putText(im_half, msg, (0, 45), cv2.FONT_HERSHEY_DUPLEX, 0.75, (109, 118, 248), 1)
            cv2.imshow('Video Stream', im_half)  # Check Detected 'Face'
            # 画像の書き出し設定
            if FrameID < 0:
                sign = 'm'
            else:
                sign = ''
            file_path = 'output/' + EventType + str(idx) + '_' + 'Event' + str(EventID) + '.' + str(abs(FrameID)) + sign + '.jpg'
            cv2.imwrite(file_path, im_half)

        # CSVの書き出し
        i = 0
        if OutputCSV == 1:
            face = cascade.detectMultiScale(im, 1.1, 3)
            if len(face) == 0:  # 顔が検出されなかった場合　
                out_list.append([videos, EventID, FrameID, TimeSTP_set, FrameNum, MovieTimeStamp, None, None, None, None, None])
            else:  # 顔が検出された場合
                for (x, y, w, h) in face:
                    if (Gaze_x - (x + w/2))**2 + (Gaze_y - (y + h/2))**2 <= (region*h/2)**2:  # 検出された顔内部に視線がある場合
                        i += 1
                        out_list.append([videos, EventID, FrameID, TimeSTP_set, FrameNum, MovieTimeStamp, i, x, y, w, h])
                else:
                    if i == 0:  # 検出された顔が全てFalseAlarmだった場合
                        out_list.append([videos, EventID, FrameID, TimeSTP_set, FrameNum, MovieTimeStamp,  None, None, None, None, None])

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
                       'GazeTimeStamp',
                       'FrameNum',
                       'MovieTimeStamp',
                       'FaceID',
                       'pos_x',
                       'pos_y',
                       'Width',
                       'Height']
    df_face.to_csv('FacePosition.csv', index=False, na_rep='NA')