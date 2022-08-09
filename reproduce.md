# 参考使用 `install.sh`，一步到位！

## 删除编译产生的文件

```shell
conda remove --name trackron --all
rm -rf dist build *.so *egg*
find -name "*pycache*" | xargs rm -rf
```

> 创建一个虚拟环境 `trackron`

```shell
conda create -n trackron python=3.9
```

> https://github.com/TAO-Dataset/tao

有关 `git+https` 的 `pkg` 的下载，只保留下载地址即可：`git+https://github.com/TAO-Dataset/tao`

不要直接使用作者提供的 `req.txt`!!! 因为里面包含了作者机器 或者 git 密钥等信息，下载会失败！！！将里面的 `==` 后面的信息去掉！

> 安装 `trackron`，注意不要使用 `install`，否者不会生成 `trackron/_C.cpython-39-x86_64-linux-gnu.so`!!! 调用处 `from trackron import _C as OPS` 就会提示找不到的错误！！！

```shell
python setup.py develop
```

# Error

`pytorch1.11.0` 对于 `trackron` 编译不通过；`1.10.1`可通过！

将 `build.py` 中的 `from timm.optim import NovoGrad...` 中的 `NovoGrad` 去掉即可，最新版本中不包含说明已经被弃用！

> 使用 `conda install` 安装 `Pytorch` 时记得把最后的 `-c conda-forge` 下载源去掉，不然会很慢！只保留 `-c pytorch` 即可！！！

`trackron` 下面的 `_C.cpython-38-x86_64-linux-gnu.so` 也是编译生成的文件，如果需要重新编译，要把这个也要删除！！！

> `git clone 128 error`

-   直接将原仓库下载下来，然后使用 `python setup.py develop/install` 安装；

更改下载源，在 `setup.py` 同级目录下创建 `setup.cfg`,

```
[easy_install]
index_url = https://pypi.tuna.tsinghua.edu.cn/simple
```

记得删除 `.git` 文件夹！

-   创建一个新的文件夹，然后运行 `git clone git+https://...` 即可，具体是什么原因导致的，不清楚！！！个人猜想还是和 `git` 有关；
-   重启网络；

## 重点

修改 `Trackron` 的源代码之后，由于作者将其封装为一个 扩展包，因此需要 重新编译 安装！将修改进行同步更新！！！

```
rm -rf build *egg* trackron/*.so

python setup.py develop
```

# TRAIN

> 单卡训练

如果不使用 `DDP` 进行训练，会报以下错误！

> 单机多卡训练

需要修改一些地方：

-   `coco` 数据集路径：`self.img_pth = os.path.join(root, 'images/{}{}/'.format(split, version))`, 去掉 `images`

在 `Trackron/trackron/data/datasets/trainsets/coco_seq.py` 中。

-   修改 `read_csv` 的方法，解决输出的警告！

`Ctrl + Shift + F` : `squeeze=True`

```python
seq_ids = pandas.read_csv(file_path,
                                header=None,
                                squeeze=True,
                                dtype=np.int64).values.tolist()

# 修改为：注意 `squeeze` 必须带 `columns` 参数！
seq_ids = pandas.read_csv(
                file_path, header=None,
                dtype=np.int64).squeeze('columns').values.tolist()

# Trackron/trackron/models/layers/position_embedding.py
dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)
# 修改为
dim_t = self.temperature**(
            2 * (torch.div(dim_t, 2, rounding_mode="floor")) /
            self.num_pos_feats)
```

-   相应参数的修改，这里有一个疑问是默认的参数都存放在 `config.py` 中的，为什么没有成功读取出来呢？

> `Trackron/trackron/solvers/build.py`

```python
# decay_rate=cfg.SOLVER.LR_SCHEDULER.DECAY_RATE,
decay_rate=0.1,
```

-   修改 `DDP` 运行脚本

```shell
# config_file=$1
config_file="configs/utt/utt.yaml"

--nproc_per_node=2  # 这里修改为对应分 GPU 数量，个人理解，还是有些出入，修改为 3 会报错，提示 bachsize=4 cannot divide by 3, 即无法合适分配！
```

> 成功运行的脚本

```python
tools/dist_train.sh

# 查看参数等信息
tensorboard --logdir=outputs  # 注意这里 outputs 里面必须包含 log.txt，否者不会成功，或者说 tensorboard 的读取路径
```

# TODO

> `SOT` 训练时候的数据集是否全用到了？怎么查看？
