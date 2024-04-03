# README

介绍此文件夹文件的含义，并配置conda环境

## Quick Install

```bash
conda create -n ENV_NAME python=311
pip install -r ./conda_env/requirements.txt
```

之后缺啥补啥。

## 文件含义

### condarc

conda配置文件（可选）。

可用于配置channels，即换源。

如要使用此配置文件，则：

```bash
mv condarc ~/.condarc
```

### rm_eng_ros.yaml

conda完整环境。

一般不能通过此文件直接安装，某些源会报错安装的包太多。只是列出笔者个人的conda环境。

注意有很多冗余的包。

若要通过此文件直接安装，则先修改文件中name和prefix字段，再：

```bash
conda env create -n ENV_NAME -f ./conda_env/rm_eng_ros.yaml
```

### requirements.txt

conda 中 pip 安装的所有包。

注意有些必要的包是使用conda安装的，即，使用miniconda时要额外安装一些包。

注意有很多冗余的包。

先创建conda环境，再用pip安装：

```bash
pip install -r ./conda_env/requirements.txt
```

### freeze_requirements.txt

用 `pip freeze` 产生的文件，便于复现。一般使用 `requirements.txt` 即可。

## conda环境配置问题

1. 若报错 `proxyerror`，则关闭系统的 proxy（代理服务器），在设置里找 proxy 并 disable
2. 建议使用 python 3.11 (3.11.6)
