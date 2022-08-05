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

- 直接将原仓库下载下来，然后使用 `python setup.py develop/install` 安装；

更改下载源，在 `setup.py` 同级目录下创建 `setup.cfg`,
```
[easy_install]
index_url = https://pypi.tuna.tsinghua.edu.cn/simple
```
记得删除 `.git` 文件夹！

- 创建一个新的文件夹，然后运行 `git clone git+https://...` 即可，具体是什么原因导致的，不清楚！！！个人猜想还是和 `git` 有关；
- 重启网络；