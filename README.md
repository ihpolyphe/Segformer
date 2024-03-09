# Segformer
画像のセマンティックセグメンテーションモデル。既存のCNNに対してTransfomerを使用して、大域的に物体を検出できる。

公式Githubは以下。

https://github.com/NVlabs/SegFormer


本リポジトリはSegformerを以下のKaggleのセグメンテーションタスクに適用。

https://www.kaggle.com/c/carvana-image-masking-challenge

## Dcoker Setup
CUDA10.2をベースにDocker Imageを作成する。

1. ImageのベースになるCUDA10.2のイメージを準備する。

cuda10.2のImageが公式からなくなってしまったので、[こちら](
https://qiita.com/dandelion1124/items/31a3452b05510097daa0)を参考に docker buildしてイメージを作成する。

2. docker buildしてImageを作成する。
```
cd Segformer
docker build -t segformer_tensorboard:latest .
```

3. コンテナを起動
```
sudo docker run --gpus all --shm-size 32G -it --rm -p 10000:8888 -p 6006:6006 -u 0 --name segformer_tensorboard segformer_tensorboard:latest
```

4. 学習を実施
```
python3 segformer.py
```

5. テストデータで推論を実施
```
python3 segformer_inference.py
```


## その他
### loss可視化
```
tensorboard --logdir="./lightning_logs/" --bind_all
http://localhost:6006
```

### Docker関連
容量をあける
```
docker images --no-trunc
docker rmi image idで消す
```
以下で容量チェック
```
docker system df
```
不要なbuild cacheは以下で削除
```
doccker builder prune
```
https://qiita.com/shione/items/dfa956a47b6632d8b3b3


## CPU使用率を制限
```
sudo cpulimit -p <process id> -l 90

sudo cpulimit -p 281597 -l 90
```

# Referrence
学習の参考
- https://blog.roboflow.com/how-to-train-segformer-on-a-custom-dataset-with-pytorch-lightning/

推論の参考
- https://cpp-learning.com/pytorch-lightning/#PyTorch_Lightning-6

推論の前処理の参考
- https://ossyaritoori.hatenablog.com/entry/2021/12/08/Pytorch%E3%81%AEPretrained_Model%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6Segmentation%E3%82%92%E8%A1%8C%E3%81%86%E5%80%8B%E4%BA%BA%E3%83%A1%E3%83%A2