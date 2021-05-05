# Darknet for CIT Brains

## 環境構築

branchをdarknet_v3に変更する．

Makefileを以下のよう変更する．

GPUあり
```shell
GPU=1
CUDNN=0
```

GPUなし
```shell
GPU=0
CUDNN=0
```

## Usage:
```shell
$ make
# make install
```
