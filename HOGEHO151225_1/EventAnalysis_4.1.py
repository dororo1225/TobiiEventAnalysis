# coding:utf-8
__author__ = 'yamamoto'
# env: Python2.7

import cv2
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
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
video = 'TB_HOGEHO151225_2.mp4'
video_name = video.split('.mp4')[0]
video_name = video_name.split('TB_')[1]

# # 視線データの指定(関数preprocessの引数)
ParName = 'HogeHoge'
RecDate = '2015-12-25'
RecName = 'Recording001'


# # イベント名
EventType = 'EC'


# # 画像のresizeの倍率
prop_resize = 1 * 1.0 / 2


# # イベント前後のtime windowの幅(FrameWindow=2: イベントの前後2フレーム)
FrameWindow = 0


# # 画像に加える回転のrotate window
interval = 5  # 回転角度の間隔
rotateNum = 9  # 回転の回数


# # 視線フィルタの引数
flagGF = 1  # 視線フィルタの有無
# region = 1  # FalseAlarmを減らす、Gazeフィルタのパラメーター

# BGRイメージを使用するかどうか
flagBGR = 1


# # Choose Face Detector
# cascade_path = "C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml"
cascade_path = "C:\opencv_300\sources\data\haarcascades\haarcascade_frontalface_alt.xml"
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

time_start = time.time()

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
if not os.path.isfile('Event_' + video_name + '.csv'):
    reader = pd.read_csv('Data Export.tsv', delimiter='\t', chunksize=1000)
    df_gaze = pd.concat((preprocess(r, ParName, RecDate, RecName) for r in reader), ignore_index=True)
    del df_gaze['Unnamed: 37']

    # フィルタリング
    df_accel = df_gaze[['Recording timestamp', 'Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']].dropna()

    N = 2
    Wn = 0.01
    b1, a1 = signal.butter(N, Wn, "low") # butterworth filter
    df_accel = pd.DataFrame({'Smooth_X': signal.filtfilt(b1, a1, df_accel['Accelerometer X'].as_matrix()),
                             'Smooth_Y':  signal.filtfilt(b1, a1, df_accel['Accelerometer Y'].as_matrix()),
                             'Smooth_Z':  signal.filtfilt(b1, a1, df_accel['Accelerometer Z'].as_matrix())},
                            index=df_accel.index)

    # df_accel['degree'] = np.rad2deg(np.arctan(df_accel['Smooth_Z'] / df_accel['Smooth_Y']))
    df_accel['Degree']\
        = np.rad2deg(np.arctan(df_accel['Smooth_Z'] / pow(df_accel['Smooth_X'] ** 2 + df_accel['Smooth_Y'] ** 2, 0.5))\
                     * (df_accel['Smooth_Y'] / abs(df_accel['Smooth_Y'])))

    # もともとのデータフレームに結合
    df_gaze = pd.concat([df_gaze, df_accel], axis=1)

    # NaNを埋める
    start = time.time()
    for index, row in df_gaze.iterrows():
        if math.isnan(row['Accelerometer X']) and index >= 2:
            df_gaze.ix[index, 'Smooth_X':'Degree'] = df_gaze.ix[(index - 1), 'Smooth_X':'Degree']
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

    # 加速度の描画
    # df_accel = df_gaze[['Recording timestamp',
    #                     'Accelerometer X', 'Accelerometer Y', 'Accelerometer Z',
    #                     'Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree']].dropna()
    # plt.subplot(3, 1, 1)
    # plt.plot(df_accel['Recording timestamp'], df_accel['Accelerometer X'], alpha=0.5)
    # plt.plot(df_accel['Recording timestamp'], df_accel['Smooth_X'], linewidth=2, alpha=0.7)
    # plt.title('Accelerometer X')
    # plt.subplot(3, 1, 2)
    # plt.plot(df_accel['Recording timestamp'], df_accel['Accelerometer Y'], alpha=0.5)
    # plt.plot(df_accel['Recording timestamp'], df_accel['Smooth_Y'], linewidth=2, alpha=0.7)
    # plt.title('Accelerometer Y')
    # plt.subplot(3, 1, 3)
    # plt.plot(df_accel['Recording timestamp'], df_accel['Accelerometer Z'], alpha=0.5)
    # plt.plot(df_accel['Recording timestamp'], df_accel['Smooth_Z'], linewidth=2, alpha=0.7)
    # plt.title('Accelerometer Z')
    # plt.tight_layout()
    # plt.show()

    # Event時のDataのみ抽出
    df_event = df_gaze[df_gaze['Event'] == EventType]  # select Event

    # csv output
    df_gaze.to_csv('Gaze_' + video_name + '.csv', index=False, na_rep='NA')
    df_event.to_csv('Event_' + video_name + '.csv', index=False, na_rep='NA')
    print 'df_gaze'
    print df_gaze.shape
    print df_gaze.head(1)
    print

    del df_gaze, df_accel
else:
    df_event = pd.read_csv('Event_' + video_name + '.csv')

# #  Eye ContactのTimwWindowとRotate Windowも含めたDataFrameをつくる(例外処理をつける)
# EyeContactの前後FrameWindow(2フレーム)を含めたDataFrameを作成
EventTimeSTP = df_event[['Project name', 'Participant name', 'Recording name', 'Recording date', 'Recording timestamp',
                         'Fixation point X', 'Fixation point Y', 'Event',
                         'Recording media width', 'Recording media height',
                         'Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree']]
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
origin = origin[['Project name', 'Participant name', 'Recording name', 'Recording date', 'Recording timestamp',
                 'EventID',
                 'FrameID',
                 'RotateID',
                 'Fixation point X',
                 'Fixation point Y',
                 'Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree',
                 'Recording media width',
                 'Recording media height']]
origin.to_csv('Origin_' + video_name + '.csv', index=False, na_rep='NA')
print len(origin.index)
del EventTimeSTP, before_upd, after_upd, before_rot, after_rot

# # Choose Face Detector
cascade = cv2.CascadeClassifier(cascade_path)
# 顔検出画像の保存ディレクトリの作成
if not os.path.isdir('output'):
    os.mkdir('output')
else:
    OutputPic = 0
    print '-----------------------------'
    print 'Folder output already exists'
    print '-----------------------------'
    print

########################
#  # Face Detection
########################

out_list = []  # Logの準備

# データをコピーしてリストへ
df_ProjName = list(origin['Project name'])
df_PartName = list(origin['Participant name'])
df_RecoName = list(origin['Recording name'])
df_RecoDate = list(origin['Recording date'])
df_TimeSTP_set = list(origin['Recording timestamp'])
df_EventID = list(origin['EventID'])
df_FrameID = map(int, list(origin['FrameID']))
df_RotateID = map(int, list(origin['RotateID']))
df_Gaze_x = list(origin['Fixation point X'])
df_Gaze_y = list(origin['Fixation point Y'])
df_Accel_x = list(origin['Smooth_X'])
df_Accel_y = list(origin['Smooth_Y'])
df_Accel_z = list(origin['Smooth_Z'])
df_Degree = list(origin['Degree'])

df_index = list(reversed(list(origin.index)))

del origin

for idx in df_index:
    # データを削除しながらイテレート
    ProjName = df_ProjName.pop()
    PartName = df_PartName.pop()
    RecoName = df_RecoName.pop()
    RecoDate = df_RecoDate.pop()
    TimeSTP_set = df_TimeSTP_set.pop()
    EventID = df_EventID.pop()
    FrameID = df_FrameID.pop()
    RotateID = df_RotateID.pop()
    Gaze_x = df_Gaze_x.pop()
    Gaze_y = df_Gaze_y.pop()
    Accel_x = df_Accel_x.pop()
    Accel_y = df_Accel_y.pop()
    Accel_z = df_Accel_z.pop()
    Degree = df_Degree.pop()

    # 画像の切り出し
    cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, TimeSTP_set)
    FrameNum = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    # cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, Frame_Num - 1)
    # FrameNum = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    MovieTimeStamp = int(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
    ret, im = cap.read()
    print int(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
    if ret:
        if flagBGR != 1:
            # グレースケール化・輝度を正規化
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # gray scale
            im = cv2.equalizeHist(im)  # 輝度を正規化
            # 画像の回転
            height, width = im.shape
        else:
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
                        region = math.sqrt(w ** 2 + h ** 2) / 2
                        if (Gaze_x * prop_resize - (x + w / 2)) ** 2 + (Gaze_y * prop_resize - (y + h / 2)) ** 2 \
                                <= region ** 2:
                            im, f_id = annotateFaceID(x, y, w, h, im_resize, f_id)
                            # print region
                # 視線フィルタによってHitがなくなった場合
                if f_id == 0:
                    noFace(im_resize)

            # 画像を描画
            # cv2.imshow('Video Stream', im_resize)
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
            if OutputPic == 0:
                file_path = None

            # 枠の描画
            if len(face_resize) == 0:  # 顔が検出されなかった場合　
                out_list.append([video, ProjName, PartName, RecName, RecDate, EventID, FrameID, RotateID,
                                 TimeSTP_set, FrameNum, MovieTimeStamp,
                                 Gaze_x, Gaze_y, Accel_x, Accel_y, Accel_z, Degree,
                                 None, None, None, None, None, file_path])
            else:  # 顔が検出された場合
                for (x, y, w, h) in face_resize:
                    if flagGF == 0:  # 視線フィルタなし
                        f_id += 1
                        out_list.append([video, ProjName, PartName, RecName, RecDate, EventID, FrameID, RotateID,
                        TimeSTP_set, FrameNum, MovieTimeStamp,
                        Gaze_x, Gaze_y, Accel_x, Accel_y, Accel_z, Degree,
                        f_id, x, y, w, h, file_path])
                    else:  # 視線フィルタあり
                        region = math.sqrt(w ** 2 + h ** 2) / 2
                        if (Gaze_x * prop_resize - (x + w / 2)) ** 2 + (Gaze_y * prop_resize - (y + h / 2)) ** 2 \
                                <= region ** 2:
                            f_id += 1
                            out_list.append([video, ProjName, PartName, RecName, RecDate, EventID, FrameID, RotateID,
                                             TimeSTP_set, FrameNum, MovieTimeStamp,
                                             Gaze_x, Gaze_y, Accel_x, Accel_y, Accel_z, Degree,
                                             f_id, x, y, w, h, file_path])

                if f_id == 0:
                    out_list.append([video, ProjName, PartName, RecName, RecDate, EventID, FrameID, RotateID,
                                     TimeSTP_set, FrameNum, MovieTimeStamp,
                                     Gaze_x, Gaze_y, Accel_x, Accel_y, Accel_z, Degree,
                                     None, None, None, None, None, file_path])

    if cv2.waitKey(10) > 0:
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()  # キャプチャー解放
cv2.destroyAllWindows()  # ウィンドウ破棄


if OutputCSV == 1:
    out_list.reverse()
    df_face = pd.DataFrame(out_list)
    print 'df_face'
    print df_face.head()
    print
    df_face.columns = ['VideoName', 'ProjectName', 'ParticipantName', 'RecordingName',  'RecordingDate',
                       'EventID', 'FrameID', 'RotateID',
                       'GazeTimeStamp', 'FrameNum', 'MovieTimeStamp',
                       'Gaze_x', 'Gaze_y', 'Accel_x', 'Accel_y', 'Accel_z', 'Degree',
                       'FaceID',
                       'pos_x',
                       'pos_y',
                       'Width',
                       'Height',
                       'Picture']
    df_face.to_csv('Face_' + video_name + '.csv', index=False, na_rep='NA')


time_elapsed = time.time() - time_start
print "elapsed_time:{0}".format(time_elapsed) + "[sec]"