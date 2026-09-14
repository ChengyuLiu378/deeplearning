# VGG块
- 一种模块化的思想，通过传入三个参数：卷积层数，输入通道数和输出通道数来构建块。块中包含若干个卷积层和激活层，在最后使用最大池化层来降低分辨率。
## 代码

```
def vgg_block(num_convs, in_channels, out_channels):

    layers = []

    for _ in range(num_convs):

        layers.append(nn.Conv2d(

            in_channels, out_channels, kernel_size=3, padding=1

        ))

        layers.append(nn.ReLU())

        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)
```
## python代码解释
`layers = []`  [[列表]]
`for _ in ...`  [[” _ “用法]]
`layers.append`  [[append]]
`*layers`  [[* layers]]
# VGG网络
- 由若干个VGG块和若干个全连接层组成。
## VGG11代码
```
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):

    conv_blks = []

    in_channels = 1

    for(num_convs, out_channels) in conv_arch:

        conv_blks.append(vgg_block(

            num_convs, in_channels, out_channels

        ))

        in_channels = out_channels

    return nn.Sequential(

        *conv_blks, nn.Flatten(),

        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),

        nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),

        nn.Dropout(0.5), nn.Linear(4096, 10)

    )

net = vgg(conv_arch)
```
# 改进
- 相较于