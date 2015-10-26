TobiiEventAnalysys_5.0
====

## Overview
Tobii Glasses2のRaw Dataを用いて解析をおこなうためのPython & Rスクリプト

## Prerequest
- Tobii Glasses2のRaw Dataをtsvファイルとして出力しておく('Data Export.csv').
- 処理したいデータの(Analysys Softwareにおける) Recording Number, Participant Name, Recording Dateをメモしておく.
- Dropboxに "AnnotationAssistant" フォルダがあることを確認しておく.
- Desktopに 'infants.csv' , 'CameraParameter.csv'があることを確認しておく.
- 処理したいデータのmp4ファイルとtsvファイルをTobiiEventAnalysys_5.0.pyのあるディレクトリに移動させておく.

## Usage
#### 0. InitializeSetting.pyを実行する.
[Process]
- 前回TobiiEventAnalysys_5.0.pyを実行した際の出力ファイルを消去, 初期化

#### 1. TobiiEventAnalysys_5.0.pyを実行する.
[Process]
 - Accelerometer dataのSmoothing, 水平面に対する頭部角度の計算
 - EventのDurationの計算
 - 画像・固視点を回転 -> 画像出力
 - Dropboxの"AnnotationAssistant" フォルダに出力画像をDataBaseとしてアップロードする


 [Output]
  - Gaze_HODEHOGE.csv : Raw Dataのcsv出力
  - Event_HOGEHOGE.csv : Event生起時のRaw Data
  - Face_HOGEHOGE.csv : AnnotationをつけるEventID, RotateIDのリスト
  - "output" フォルダ : 出力画像


#### 2. "AnnotationAssistant"フォルダ → "FaceClipper"フォルダのStart.batを実行する.
[Process]
 - 出力画像について, 手動でAnnotationをつける
 - Dropbox上の 'HOGEHOGE.db' のデータを更新する


#### 3. "AnnotationAssistant"フォルダ → "DBOutput"フォルダのDBOutput.pyを実行する.
[Process]
 - 'HOGEHOGE.db'のデータをcsv出力する


 [Output]
 - 'AnnotationAssistant/DBOutput/AnnotationData.csv'

#### 4. preprocess.Rmdを実行する.
[Process]
 - 'AnnotationAssistant/DBOutput/AnnotationData.csv'を読み込む
 - 'Face_HOGEHOGE.csv' を読み込み, Heightの列を手動コードしたデータで更新する
 - 手動コードした顔のpixel上のサイズから, 水平距離を推定
 - データ更新後の 'Face_HOGEHOGE.csv' 由来のデータをcsv出力( 'Face_HOGEHOGE,csv' )
 - 各Sessionの基本情報をcsv出力( 'Summary_HOGEHOGE.csv' )

 [Output]
 - Face_HOGEHOGE.csv :  データ更新後のRaw Dataのcsv出力
 - Session_HOGEHOGE.csv    : 各Sessionの基本情報

## Author

[dororo1225](https://github.com/dororo1225)
