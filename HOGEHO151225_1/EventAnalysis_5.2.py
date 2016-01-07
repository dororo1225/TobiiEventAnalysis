# coding:utf-8
__author__ = 'yamamoto'
# env: Python2.7

import cv2
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os
import sys
import shutil
import math
import time
import sqlite3
import datetime

import pandas.tseries.offsets as offsets

################################################
#  初期入力
################################################

# # 出力の指定
calcDuration = 1  # Whether caliculate Event Duration
OutputPic = 1    # Whether Output Movie (1:Output / 0:Don't Output)
OutputCSV = 1  # Whether OutputCSV (1:Output / 0:Don't Output)

# # 入力の指定(Video Input)
video = 'TB_HOGEHO151225_1.mp4'
video_name = video.split('.mp4')[0]
video_name = video_name.split('TB_')[1]
obs_id = video_name[0:12]
video_num = int(video_name[-1])

# # Tobiiデータの指定
TBdata = 'Data Export.tsv'

# # イベント名
EventType = 'EC'


# # 画像のresizeの倍率
prop_resize = 1 * 1.0 / 2


# # 画像に加える回転のrotate window
interval = 10  # 回転角度の間隔
rotateNum = 3      # 回転の回数

# # DataBase作成のためのタグ (0でOK)
copyFiles = 0
formatDB = 0

# errorフラッグ
error = 0

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


# DBのフォーマット関数
def format_db(dpath, fpath):
    conn = sqlite3.connect(dpath)
    # samplesテーブル
    sql = u"""
    create table samples (
    id integer primary key autoincrement,
    filepath text unique not null,
    x integer not null default 0,
    y integer not null default 0,
    width integer not null default 0,
    height integer not null default 0,
    status integer not null default 0,
    teacher text default 'no name',
    updated_date timestamp default (datetime('now', 'localtime'))
    );
    """
    conn.execute(sql)

    sql = u"""
    create index id_idx on samples(id);
    """
    conn.execute(sql)

    sql = u"""
    create table progress (
    pos integer primary key,
    total integer not null
    );
    """
    conn.execute(sql)

    sql = u"""
    create index pos_idx on progress(pos);
    """
    conn.execute(sql)

    for item in fpath:
        sql = u"""insert into samples(filepath, status) values (?, 100); """
        conn.execute(sql, (unicode(item, sys.getfilesystemencoding()),))

    sql = u"""insert into progress values (?,?);"""
    fnum = (0, len(fpath))
    conn.execute(sql, fnum)
    conn.commit()
    conn.close()

################################################
#  メイン処理
################################################

time_start = time.time()

while True:

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
    desktop_path = os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop"
    df_obs = pd.read_csv(desktop_path + '\\Observation.csv')
    ParName = list(df_obs[df_obs['ObsID'] == obs_id]['Mother'])[0]
    RecDate = list(df_obs[df_obs['ObsID'] == obs_id]['RecDate'])[0]
    RecDate = datetime.datetime.strptime(RecDate, '%Y/%m/%d').strftime('%Y-%m-%d')
    RecName = list(df_obs[df_obs['ObsID'] == obs_id]['RecName'])[0][1:-1].replace("'", "").replace(" ", "").split(",")[video_num - 1]

    # # 視線データの読み込み
    RecName = RecName if 'RecName' in locals() else 'Recording'
    if (os.path.isfile('Gaze_' + video_name + '.csv')) or (os.path.isfile('Event_' + video_name + '.csv')):
        print '-------------------------------------------'
        print 'Gaze File or Event File has already exists'
        print '-------------------------------------------'
        print
        break

    reader = pd.read_csv(TBdata, delimiter='\t', chunksize=1000)
    df_gaze = pd.concat((preprocess(r, ParName, RecDate, RecName) for r in reader), ignore_index=True)
    del df_gaze['Unnamed: 37']

    # Event時のDataのみ抽出
    df_event = df_gaze[df_gaze['Event'] == EventType]  # select Event

    # Eventの生起時刻を計算
    df_event = df_event.assign(Recording_stime=pd.to_datetime(df_event['Recording start time'], format='%H:%M:%S'))
    event_time = []
    for idx in df_event.index:
        evt = df_event.ix[idx, 'Recording_stime'] + offsets.Milli(df_event.ix[idx, 'Recording timestamp'])
        event_time.append(evt.strftime("%H:%M"))
    df_event = df_event.assign(EventTime=event_time)

    print df_event

    # EventのDurationを計算
    df_end = df_gaze[df_gaze['Event'] == 'EC_End']
    if calcDuration == 1:
        if df_event.shape[0] != df_end.shape[0]:
            error = 1
            print '-------------------------------------'
            print 'Number of Event is'
            print 'different from Number of Event End'
            print 'Check Tobii Data Again'
            print '-------------------------------------'
            print

        EventInterval = df_event['Recording timestamp'].as_matrix()[1:df_event.shape[0]] -\
                            df_end['Recording timestamp'].as_matrix()[0:df_event.shape[0]-1]
        if (EventInterval < 1000).any():
            error = 1
            print '-------------------------------------'
            print 'Intervals between Events are'
            print 'Over threshold'
            print 'Check Tobii Data Again'
            print df_end['Recording timestamp'].as_matrix()[0:df_event.shape[0]-1] [EventInterval < 1000], 'ms'
            print '-------------------------------------'

        if error == 1:
            break

        df_end.index = df_event.index
        df_event = df_event.assign(Duration=df_end['Recording timestamp'] - df_event['Recording timestamp'])
    else:
        df_event = df_event.assign(Duration=[None] * df_end.shape[0])

    # フィルタリング
    df_accel = df_gaze[['Recording timestamp', 'Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']].dropna()
    N = 2
    Wn = 0.01
    b1, a1 = signal.butter(N, Wn, "low") # butterworth filter
    df_accel = pd.DataFrame({'Smooth_X': signal.filtfilt(b1, a1, df_accel['Accelerometer X'].as_matrix()),
                             'Smooth_Y':  signal.filtfilt(b1, a1, df_accel['Accelerometer Y'].as_matrix()),
                             'Smooth_Z':  signal.filtfilt(b1, a1, df_accel['Accelerometer Z'].as_matrix())},
                            index=df_accel.index)
    df_accel['Degree'] = \
        np.rad2deg(np.arctan(df_accel['Smooth_Z'] / pow(df_accel['Smooth_X'] ** 2 + df_accel['Smooth_Y'] ** 2, 0.5)) \
                   * df_accel['Smooth_Y'] / abs(df_accel['Smooth_Y']))

    # もともとのデータフレームに結合
    df_gaze = pd.concat([df_gaze, df_accel], axis=1)

    # Nanを前の値で置換
    df_gaze[['Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree']] \
        = df_gaze[['Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree']].fillna(method='ffill')

    # 加速度の描画
    df_accel = df_gaze[['Recording timestamp',
                        'Accelerometer X', 'Accelerometer Y', 'Accelerometer Z',
                        'Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree']].dropna()
    plt.subplot(3, 1, 1)
    plt.plot(df_accel['Recording timestamp'], df_accel['Accelerometer X'], alpha=0.5)
    plt.plot(df_accel['Recording timestamp'], df_accel['Smooth_X'], linewidth=2, alpha=0.7)
    plt.title('Accelerometer X')
    plt.subplot(3, 1, 2)
    plt.plot(df_accel['Recording timestamp'], df_accel['Accelerometer Y'], alpha=0.5)
    plt.plot(df_accel['Recording timestamp'], df_accel['Smooth_Y'], linewidth=2, alpha=0.7)
    plt.title('Accelerometer Y')
    plt.subplot(3, 1, 3)
    plt.plot(df_accel['Recording timestamp'], df_accel['Accelerometer Z'], alpha=0.5)
    plt.plot(df_accel['Recording timestamp'], df_accel['Smooth_Z'], linewidth=2, alpha=0.7)
    plt.title('Accelerometer Z')
    plt.tight_layout()
    # plt.show()
    plt.savefig(video_name + '.png')
    # print len(df_gaze) - df_gaze.count()

    # ##############################
    # Eventの部分だけ抽出/df_eventと合体
    df_event = pd.concat([df_event, df_gaze[['Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree']]],
                        axis=1, join_axes=[df_event.index])

    #  csv output
    # df_gaze.to_csv('Gaze_' + video_name + '.csv', index=False, na_rep='NA')
    del df_gaze, df_accel, df_end, df_event['Recording_stime']

    # Eye ContactEventにRotate Windowも含めたDataFrameをつくる(例外処理をつける)
    df_event = df_event.assign(EventID=range(1, df_event.shape[0] + 1))
    df_event = df_event[['Project name', 'Participant name', 'Recording name', 'Recording date',
                         'EventTime', 'Recording timestamp', 'Event', 'EventID',
                         'Fixation point X',
                         'Fixation point Y',
                         'Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree', 'Duration',
                         'Recording media width',
                         'Recording media height']]
    df_event.index = range(0, df_event.shape[0])
    print '------------------------------------------------------------------------------------'
    print 'Event_' + video_name + '.csv'
    print df_event
    print
    df_event.to_csv('Event_' + video_name + '.csv', index=False, na_rep='NA')

    # # 各フレームに回転を加えた画像を含めたDatFrameを作成
    ID_rotate = pd.DataFrame({'RotateID': np.zeros(df_event.shape[0])})
    before_rot = pd.concat([df_event, ID_rotate], axis=1)
    after_rot = pd.concat([df_event, ID_rotate], axis=1)
    df_event = pd.concat([df_event, ID_rotate], axis=1)
    for var in range(1, rotateNum + 1):
        before_rot[['RotateID']] -= interval
        after_rot[['RotateID']] += interval
        df_event = pd.concat([df_event, before_rot, after_rot])

    df_event = df_event.sort_values(['Recording timestamp', 'EventID', 'RotateID'])
    df_event.index = range(0, df_event.shape[0])

    df_event = df_event[['Project name', 'Participant name', 'Recording name', 'Recording date',
                         'EventTime', 'Recording timestamp', 'Event', 'EventID', 'RotateID',
                         'Fixation point X',
                         'Fixation point Y',
                         'Smooth_X', 'Smooth_Y', 'Smooth_Z', 'Degree', 'Duration',
                         'Recording media width',
                         'Recording media height']]

    # 画像の保存ディレクトリの作成
    if os.path.isdir('output'):
        print '-----------------------------'
        print 'Folder output already exists'
        print '-----------------------------'
        print

    os.mkdir('output')  # 画像の仮出力フォルダを作成
    out_list = []  # Logの準備

    # データをコピーしてリストへ
    df_ProjName = list(df_event['Project name'])
    df_PartName = list(df_event['Participant name'])
    df_RecoName = list(df_event['Recording name'])
    df_RecoDate = list(df_event['Recording date'])
    df_EventTime = list(df_event['EventTime'])
    df_TimeSTP_set = list(df_event['Recording timestamp'])
    df_Event = list(df_event['Event'])
    df_EventID = list(df_event['EventID'])
    df_RotateID = map(int, list(df_event['RotateID']))
    df_Gaze_x = list(df_event['Fixation point X'])
    df_Gaze_y = list(df_event['Fixation point Y'])
    df_Accel_x = list(df_event['Smooth_X'])
    df_Accel_y = list(df_event['Smooth_Y'])
    df_Accel_z = list(df_event['Smooth_Z'])
    df_Degree = list(df_event['Degree'])
    df_Duration = list(df_event['Duration'])

    df_index = list(reversed(list(df_event.index)))
    del df_event, before_rot, after_rot

    for idx in df_index:
        # データを削除しながらイテレート
        ProjName = df_ProjName.pop()
        PartName = df_PartName.pop()
        RecoName = df_RecoName.pop()
        RecoDate = df_RecoDate.pop()
        EventTime = df_EventTime.pop()
        TimeSTP_set = df_TimeSTP_set.pop()
        Event = df_Event.pop()
        EventID = df_EventID.pop()
        RotateID = df_RotateID.pop()
        Gaze_x = df_Gaze_x.pop()
        Gaze_y = df_Gaze_y.pop()
        Accel_x = df_Accel_x.pop()
        Accel_y = df_Accel_y.pop()
        Accel_z = df_Accel_z.pop()
        Degree = df_Degree.pop()
        Duration = df_Duration.pop()

        # 画像の切り出し
        cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, TimeSTP_set)
        FrameNum = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        MovieTimeStamp = int(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
        ret, im = cap.read()

        if ret:
            height, width, cha = im.shape
            M = cv2.getRotationMatrix2D((width / 2, height / 2), RotateID, 1)
            im = cv2.warpAffine(im, M, (width, height))

            # 視線の回転
            Gaze_x, Gaze_y = rotateGaze(Gaze_x, Gaze_y, width / 2, height / 2, RotateID)

            # 画像のリサイズ
            im_resize = cv2.resize(im, (int(im.shape[1] * prop_resize), int(im.shape[0] * prop_resize)))

            # 出力画像名
            if RotateID < 0:
                sign_r = 'm'
            else:
                sign_r = ''
            file_path = 'output/' + 'Event' + str(idx).zfill(4) + '_' + video_name + '_' + Event + '.' +\
                        str(EventID) + '.' + str(abs(RotateID)) + sign_r + '.jpg'

            # アノテーションを画像に付加して描画
            if OutputPic == 1:
                # 画像の左上にEventID, FrameID, RotateID, RecName, RecDateを記載する
                cv2.rectangle(im_resize, (0, 0), (145, 85), (50, 50, 50), -1)
                title_top = Event + ': ' + str(EventID) + ', ' + str(RotateID)
                cv2.putText(im_resize, title_top, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                title1 = RecDate + ' ' + EventTime
                title2 = RecName
                cv2.putText(im_resize, title1, (0, 40), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(im_resize, title2, (0, 60), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)
                T_set = TimeSTP_set / (1000 * 60 * 60)
                M_set = (TimeSTP_set - T_set * 1000 * 60 * 60) / (1000 * 60)
                S_set = (TimeSTP_set - T_set * 1000 * 60 * 60 - M_set * 1000 * 60) * 1.0 / 1000
                title3 = str(T_set).zfill(2) + ':' + str(M_set).zfill(2) + ':' + str(round(S_set, 3)).zfill(6)
                cv2.putText(im_resize, title3, (0, 80), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)

                # 画像上に視点データを描画する
                cv2.circle(im_resize, (int(Gaze_x * prop_resize), int(Gaze_y * prop_resize)), 8, (109, 118, 248), 1)

                # 画像を描画
                cv2.imshow('Video Stream', im_resize)
                # キー入力待機
                # cv2.waitKey(0)
                # 画像の書き出し
                cv2.imwrite(file_path, im_resize)

            # CSVの書き出し
            if OutputCSV == 1:
                    out_list.append([video, ProjName, PartName, RecName, RecDate,
                                     EventTime, TimeSTP_set,
                                     Event, EventID, RotateID,
                                     FrameNum, MovieTimeStamp,
                                     Gaze_x, Gaze_y, Accel_x, Accel_y, Accel_z, Degree, Duration,
                                     None, None, None, None, file_path])

        if cv2.waitKey(10) > 0:
            cap.release()
            cv2.destroyAllWindows()
            break
    else:
        if OutputCSV == 1:
            out_list.reverse()
            df_face = pd.DataFrame(out_list)
            print 'df_face'
            print df_face.head()
            print
            df_face.columns = ['VideoName', 'ProjectName', 'ParticipantName', 'RecordingName',  'RecordingDate',
                               'EventTime', 'GazeTimeStamp',
                               'Event', 'EventID', 'RotateID',
                               'FrameNum', 'MovieTimeStamp',
                               'Gaze_x', 'Gaze_y', 'Accel_x', 'Accel_y', 'Accel_z', 'Degree', 'Duration',
                               'pos_x', 'pos_y', 'Width', 'Height', 'Picture']
            df_face.to_csv('Face_' + video_name + '.csv', index=False, na_rep='NA')

        if OutputPic == 1 and OutputCSV == 1:
            copyFiles = 1
    break

cap.release()  # キャプチャー解放
cv2.destroyAllWindows()  # ウィンドウ破棄

time_elapsed = time.time() - time_start
print "elapsed_time:{0}".format(time_elapsed) + "[sec]"
print 'Finished!!!'
print

################################################
# DBのフォーマット
################################################

if copyFiles == 1:
    if os.path.isdir('AnnotationAssistant\\FaceClipper\\static\\images'):
        print '------------------------------------'
        print 'Check inside folder "FaceClipper/static"'
        print '------------------------------------'
        print
    else:
        os.mkdir('AnnotationAssistant\\FaceClipper\\static\\images')
        for idx in df_face.index:
            src = df_face.ix[idx, 'Picture']
            src = src.replace('/', '\\')
            shutil.copy(src, 'AnnotationAssistant\\FaceClipper\\static\\images')
        else:
            formatDB = 1

print os.getcwd().split('\\')[-1]

if copyFiles == 1 and formatDB == 1:
    obs_id = os.getcwd().split('\\')[-1]

    # USB coding用の空フォルダとtxtファイルをつくる.
    file_name = obs_id + '.txt'
    str = """cd App\Lib\site-packages
    set PYTHONPATH=%cd%
    cd ../../../
    start http://127.0.0.1:5000/clipper
    App\python.exe """ + obs_id + """/clipper.py
    pause"""
    # USB coding用 txtファイルの作成
    f = open('AnnotationAssistant\\FaceClipper\\static\\' + file_name, 'w') # 書き込みモードで開く
    f.write(str) # 引数の文字列をファイルに書き込む
    f.close() # ファイルを閉じる
    # USB codin用 空フォルダの作成
    os.mkdir('AnnotationAssistant\\FaceClipper\\static\\' + obs_id)

    # DBにアップロードする画像pathを取得
    os.chdir('AnnotationAssistant/FaceClipper/static/images')
    files = os.listdir(os.getcwd())  # ディレクトリ内のファイル名を取得
    # 画像データのpathのリストを作成
    file_path = []
    for var in files:
        file_path.append('static/images/' + var)
    # file_num = (0, len(file_path))

    # #############################################意味ある？

    os.chdir('../../../DBFormat')  # デバッグ初期の作業ディレクトリに戻す
    print os.getcwd()

    # DBファイルの場所としてDropboxのpathを取得
    db_name = obs_id + '.db'
    # data_path = os.getenv("HOMEDRIVE") + \
    #                     os.getenv("HOMEPATH") +  \
    #                     "\\Dropbox\\AnnotationAssistant\\" + db_name
    data_path = os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Dropbox\\teach_db\\" + db_name # 練習用DBフォルダ
    print data_path

    if os.path.isfile(data_path):
        print "---------------------------------------------"
        print db_name + " has already existed in Dropbox."
        print "Check contents of 'samples.db'"
        print "---------------------------------------------"
    else:
        format_db(data_path, file_path)

        # sample.dbの中身を取得
        print "-------------------------------------------------------"
        print "Check content of " + db_name
        print data_path
        print "-------------------------------------------------------"
        connector = sqlite3.connect(data_path)
        cursor = connector.cursor()

        # samplesテーブルのレコードを取得
        cursor.execute("select * from samples")
        for var in range(0, 5):  # 先頭5行のレコードを表示
            row = cursor.fetchone()
            print row
        # for row in cursor:  # すべてのレコードを表示する.
        # print row

        # progressテーブルのレコードを取得
        cursor.execute("select * from progress")
        for row in cursor:
            print row

        # sample.dbを閉じる
        connector.close()