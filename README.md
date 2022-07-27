# Gym-Test

##前置き
自作環境で行う前にOpenAI Gymで試しただけなのでコードが汚いです笑  
何か間違いがあれば教えてください。。。  
参考に（ほぼコピペ）したのはめんだこさんのサイト。違いはtensorflowをpytrochに書きなおしてるくらい。

##実装したゲームとアルゴリズム
* CartPole-v0　([Ape-X DQN][1])
* BreakoutDeterministic-v4 (Ape-X DQN)
* Pendulum-v1 ([PPO][2]・DDPG)

##バージョンの確認
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
> capacity : リプレイメモリーのサイズ　→　2の累乗なのはApe-Xの優先順序を完全2分木で保存しているから(default: 2 ** 21)  
> epsilon : 探索率(default: 0.5) →　$\epsilon^{1+i/(N-1)\alpha}$  
> eps_alpha : 探索率(default: 7)  
> advanced  
> td_epsilon  
> interval  
> update  
> target_update  
> min_replay  
> local_cycle  
> num_minibatch  
n_frame  

python Breakout.py とかでできんじゃないか。  

バージョンとか諸々忘れた。

[1]:https://horomary.hatenablog.com/entry/2021/03/02/235512
[2]:https://horomary.hatenablog.com/entry/2020/10/22/234207
