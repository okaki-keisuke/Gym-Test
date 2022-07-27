# Gym-Test

## 前置き
自作環境で行う前にOpenAI Gymで試しただけなのでコードが汚いです笑  
何か間違いがあれば教えてください。。。  
参考に（ほぼコピペ）したのはめんだこさんのサイト(参考文献)。違いはtensorflowをpytrochに書きなおしてるくらい。

## 実装したゲームとアルゴリズム
* CartPole-v0　([Ape-X DQN][1])
* BreakoutDeterministic-v4 (Ape-X DQN)
* Pendulum-v1 ([PPO][2]・DDPG)

## バージョンの確認
> gym : 0.24.1   
> pytorch : 1.11.0+cu113 (もちろんcuda 11.3)  
> ray : 1.13.0  
> numpy : 1.20.3  
> tensorboard : 2.6.0  

## parserの説明(Ape-Xだけかな？)
> graph : グラフを作成するか(default: False)  
> save : modelを保存するか(default: False)  
> gamma : 報酬の割引率(default: 0.99)  
> batch : バッチサイズ(default: 512)  
> capacity : リプレイメモリーのサイズ(default: 2 ** 21)　→　2の累乗なのはApe-Xの優先順序を完全2分木で保存しているから  
> epsilon : 探索率(default: 0.5) → $\epsilon^{1+\frac{i}{N-1}\alpha}$の$\epsilon$  
> eps_alpha : 探索率(default: 7) → $\epsilon^{1+\frac{i}{N-1}\alpha}$の$\alpha$  
> advanced : advance stepの数(default: 3)  
> td_epsilon : メモリに優先順序を付ける時のバイアス(default: 0.001)  
> interval : テストの間隔(default: 10)  
> update : 1updatesの回数(default: 5000 updates)　→　1updates = 16 minibatch upate = 16 update  
> target_update : target Networkの更新頻度(default: 2400 update)  
> min_replay : 学習を始める時のリプレイメモリーの値(default: 50000)  
> local_cycle : Actorが画像をリプレイメモリーに投げる時のデータ数(default: 100)  
> num_minibatch : 1updatesで更新するminibatchの数(default: 16)  
> n_frame : stateで使う直近の画像の数(default: 4)    

## 実行方法
### CartPole
```
python Cartpole.py --graph --save
```
### Breakout
```
python Breakout.py --graph --save
```
### Pendulum
```
python ppo.py
or
python ddpg.py
```

## 注意
モデルの保存先は変えないといけないはずなのでよさげなところ指定してほしい。
PPOの実装は無理やりpytorchに直した感じだからいろいろ変えてみてほしい。もう少しうまいやり方がありそう。

## 参考文献
[Distributed Prioritized Experience Replay][3]  
[Proximal policy optimization algorithms][4]  
[Deterministic Policy Gradient Algorithms][5]  

[1]:https://horomary.hatenablog.com/entry/2021/03/02/235512
[2]:https://horomary.hatenablog.com/entry/2020/10/22/234207
[3]:https://arxiv.org/abs/1803.00933
[4]:https://arxiv.org/abs/1707.06347
[5]:http://proceedings.mlr.press/v32/silver14.html
