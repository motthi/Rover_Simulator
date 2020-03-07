# Rover Simulator

## 概要

2輪ローバーのシミュレータ  

## 動作環境
以下の環境で実行できたことを確認済み  
- Python 3.7.4  
- Jupyter Notebook 1. 2. 4

Jupyter Labではアニメーションの再生が出来なかった。


## 使い方

- リポジトリをクローン  
`git clone https://github.com/pasises/Rover_Simulator`  

- Jupyter Notebookを起動  
`Jupyter Notebook`

- ブラウザで http://localhost:8888 にアクセスする  
- Simulator.ipynbを開き，Kernel -> Restart & Run Allでシミュレーションを実行する  

- Agentを自分で定義しなおし，検証する

## クラスについて
### IdealRobot.py 
シミュレーションを行うにあたって最低限必要なクラスを定義  
ロボットやセンサは理想的  
- Worldクラス  
シミュレーション環境を定義  
	- time_span : シミュレーション実行時間 [s]
	- time_interval : ステップ時間 [s]
	- debug=False : デバック設定（True:デバッグ，False：シミュレーション）
	- goal_pos=[None, None] : ゴール座標， [m] [m]（2要素配列）
	- map_size=[100, 100] : マップサイズ， [m] [m]（2要素配列）

- IdealRobotクラス  
理想的に行動するロボットを定義  
	- pose : 初期位置，[m] [m]（2要素配列）
	- agent=None : ロボットに載せるエージェント（Agentクラス）
	- sesor=None : ロボットに載せるセンサ（IdealSensorクラス） 
	- color='black' : 描画上のロボットの色
- IdealSensorクラス  
理想的な値を返すセンサを定義  
引数なし

- Agentクラス  
エージェントを定義  
	- goal_pos=[0, 0] : ゴール座標 [m], [m]（2変数配列）
	- motor=[0, 0] : モータ制御値 [%], [%]（2変数配列， 一軸制御したい場合のみ）

#### Robot.py	
より現実に即したシミュレーションを行うためのクラスを定義  
- Robotクラス  
ロボットを定義  
IdealRobotクラスを継承
	- pose : 初期位置 [m], [m] （2変数配列）
	- agent=None : ロボットに載せるエージェント（Agentクラス）
	- sensor=None : ロボットに載せるセンサ（Sensorクラス）
	- color='black' : 描画上のロボットの色
	- noise_per_meter=5 : 向きが変わりうる距離 [m]
	- noise_std=math.pi/50 : 向きが変わる際に変化する角度の標準偏差 [rad]
	- bias_rate_stds=(0.1, 0.1): 走行時のバイアス [m/s], [rad/s]（2要素タプル）
	- expected_stuck_time=1e100 : スタックが生じるまでの時間 [s]
	- expected_escape_time=1e-100 : スタックから脱出するまでの時間 [s]

- Sensorクラス  
センサを定義  
Sensorクラスを継承
	- distance_noise_rate=0.1 : GPS誤差の標準偏差 [m]
	- direction_noise_rate=0.1 : 地磁気センサ誤差の標準偏差 [deg]
	- time_interval=None : ステップ時間 [s]（引数に入れると1秒ごとにGPSが更新されるようになる）

### Agentの実装方法
エージェントを変えたい場合はAgentクラスを継承する新しいクラスを定義する．  
actionメソッドをオーバーライドすることで複雑な制御を実行できる．  
定義したクラスのコンストラクタ引数にゴール座標を加えることで，ゴールへ移動するエージェントを実装できるようになる（参考：Simulator.ipynb [2]）．

## アニメーションの保存
Simulator.ipynb[6]の最後に書かれている以下のコード  
`#world.ani.save('anm.gif, 'writer='pillow', fps=10)`  
のコメントアウトを外すことでアニメーションを作業ディレクトリに保存することができる．  

## 参考文献
上田隆一　『詳解　確率ロボティクス　Pythonによる基礎アルゴリズムの実装』 （講談社、2019）

