# 参考使用 `install.sh`，一步到位

## 删除编译产生的文件

```shell
conda remove --name trackron --all
rm -rf dist build trackron/*.so *egg*
find -name "*pycache*" | xargs rm -rf
```

> 创建一个虚拟环境 `trackron`

```shell
conda create -n trackron python=3.9
```

> <https://github.com/TAO-Dataset/tao>

有关 `git+https` 的 `pkg` 的下载，只保留下载地址即可：`git+https://github.com/TAO-Dataset/tao`

不要直接使用作者提供的 `req.txt`!!! 因为里面包含了作者机器 或者 git 密钥等信息，下载会失败！！！将里面的 `==` 后面的信息去掉！

> 安装 `trackron`，注意不要使用 `install`，否者不会生成 `trackron/_C.cpython-39-x86_64-linux-gnu.so`!!! 调用处 `from trackron import _C as OPS` 就会提示找不到的错误！！！

```shell
python setup.py develop
```

## Error

`pytorch1.11.0` 对于 `trackron` 编译不通过；`1.10.1`可通过！

将 `build.py` 中的 `from timm.optim import NovoGrad...` 中的 `NovoGrad` 去掉即可，最新版本中不包含说明已经被弃用！

> 使用 `conda install` 安装 `Pytorch` 时记得把最后的 `-c conda-forge` 下载源去掉，不然会很慢！只保留 `-c pytorch` 即可！！！

`trackron` 下面的 `_C.cpython-38-x86_64-linux-gnu.so` 也是编译生成的文件，如果需要重新编译，要把这个也要删除！！！

> `git clone 128 error`

- 直接将原仓库下载下来，然后使用 `python setup.py develop/install` 安装；

更改下载源，在 `setup.py` 同级目录下创建 `setup.cfg`,

```yaml
[easy_install]
index_url = https://pypi.tuna.tsinghua.edu.cn/simple
```

记得删除 `.git` 文件夹！

- 创建一个新的文件夹，然后运行 `git clone git+https://...` 即可，具体是什么原因导致的，不清楚！！！个人猜想还是和 `git` 有关；
- 重启网络；

## 重点

修改 `Trackron` 的源代码之后，由于作者将其封装为一个 扩展包，因此需要 重新编译 安装！将修改进行同步更新！！！

```bash
rm -rf build *egg* trackron/*.so

python setup.py develop
```

## TRAIN

> 单卡训练

如果不使用 `DDP` 进行训练，会报以下错误！

> 单机多卡训练

### 需要修改一些地方

> `scheduler` 修改：[Swin transformer TypeError: **init**() got an unexpected keyword argument ‘t_mul‘\_3DYour 的博客-CSDN 博客](https://blog.csdn.net/abc1831939662/article/details/123477853) 直接注释 提示缺少的参数

```python
lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_iters,
            # XBL comment,
            # t_mul=cfg.SOLVER.LR_SCHEDULER.LR_CYCLE_MUL,
            lr_min=cfg.SOLVER.LR_SCHEDULER.LR_MIN,
            # XBL comment,
            # decay_rate=cfg.SOLVER.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=cfg.SOLVER.LR_SCHEDULER.WARMUP_LR,
            warmup_t=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
            cycle_limit=cfg.SOLVER.LR_SCHEDULER.LR_CYCLE_LIMIT,
            t_in_epochs=False,
            noise_range_t=noise_range,
            noise_pct=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_PCT,
            noise_std=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_STD,
            noise_seed=cfg.SEED,
        )
```

- `coco` 数据集路径：`self.img_pth = os.path.join(root, 'images/{}{}/'.format(split, version))`, 去掉 `images`

在 `Trackron/trackron/data/datasets/trainsets/coco_seq.py` 中。

- 修改 `read_csv` 的方法，解决输出的警告！

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

- 相应参数的修改，这里有一个疑问是默认的参数都存放在 `config.py` 中的，为什么没有成功读取出来呢？

> `Trackron/trackron/solvers/build.py`

```python
# decay_rate=cfg.SOLVER.LR_SCHEDULER.DECAY_RATE,
decay_rate=0.1,
```

- 修改 `DDP` 运行脚本

```shell
# config_file=$1
config_file="configs/utt/utt.yaml"

--nproc_per_node=2  # 这里修改为对应分 GPU 数量，个人理解，还是有些出入，修改为 3 会报错，提示 bachsize=4 cannot divide by 3, 即无法合适分配！
```

修改 `utt.yaml` 中的配置文件，将 `batchsize` 调整为 `3` 的倍数，实验中调整为 `24`。

> 成功运行的脚本

```python
tools/dist_train.sh

# 查看参数等信息
tensorboard --logdir=outputs  # 注意这里 outputs 里面必须包含 log.txt，否者不会成功，或者说 tensorboard 的读取路径
```

### 运行时配置（看情况是否进行修改）

> 1.是否进行可视化展示

`trackron/config/defaults.py`,

```python
TRACKER.VISUALIZATION
```

> 2.修改 `SOT` 图片大小，默认设置在配置文件中 `352`

`trackron/config/data_configs.py`, 320

`configs/utt/utt.yaml`, 352

直接搜索 `SEARCH.SIZE`，然后对 `SOT` 的搜索区域大小进行修改，初步考虑将 `SEARCH.SIZE` 替换为 `img_info` 中的 `w, h`。

最终修改的地方：`trackron/data/processing/base.py`，将图片裁剪部分操作去除即可；默认的训练方式是 `SOT`。

## Q & A

> Q: `SOT` 训练时候的数据集是否全用到了？怎么查看？

A: 在配置文件 `utt.yaml` 或者 终端日志 输出中可以看到，对与 `SOT` 模式使用的训练数据集为 `SOT & coco` 数据集，并未使用到 `MOT` 数据集进行混合训练！

> Q: 是否可以在 `SOT & MOT` 之间进行切换？

A: 可以。具体实现是设置经过 30 个间隔，在两种模式之间进行切换。`trackron/trackers/tracking_actor.py` 中注释的部分：`self.tracker.switch_tracking_mode(image, info)`

## 成功运行

```bash
python tools/train_net.py --config-file "configs/utt/utt.yaml" --config-func utt
```

## RESUME

个人猜测从 `last_checkpoint` 进行加载，而非 `.pth` 文件！！！

1. 修改 `utt.yaml` 中的 `weights`;

```yaml
# WEIGHTS: "https://download.pytorch.org/models/resnet50-19c8e357.pth"
WEIGHTS: "/home/guest/XieBailian/proj/Trackron/outputs/last_checkpoint"
```

### `ERROR` 解决

- `trackron/checkpoint/tracking_checkpoint.py`: `func save()`
- `/home/guest/anaconda3/envs/trackron/lib/python3.8/site-packages/fvcore/common/checkpoint.py`: `meta` 官方的 `fvcore`

> 参考链接 1：[Py 之 fvcore：fvcore 库的简介、安装、使用方法之详细攻略\_一个处女座的程序猿的博客-CSDN 博客\_fvcore 安装](https://blog.csdn.net/qq_41185868/article/details/103881195)
> 参考链接 2：[How to resume training from last checkpoint? · Issue #148 · facebookresearch/detectron2](https://github.com/facebookresearch/detectron2/issues/148)
