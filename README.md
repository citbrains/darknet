# Darknet for CIT Brains

## 環境構築

各自のPCで環境構築する方法を以下に示す．

branchをdarknet_v3に変更する．

Makefileを以下のよう変更する．

* GPUあり
```shell
GPU=1
CUDNN=0
```
CUDNN=1でビルドできるなら，それが望ましい．

* GPUなし
```shell
GPU=0
CUDNN=0
```

## Usage:
```shell
$ make
# make install
```
