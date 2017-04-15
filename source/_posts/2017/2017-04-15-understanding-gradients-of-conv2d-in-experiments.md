---
title: 理解conv2d及其梯度的计算过程
date: 2017-04-15 11:48:56
categories:
- Machine Learning
tags:
- tensorflow
- AI
- Machine Learning
- Deep Learning
---

在当前深度学习领域中，卷积神经网络在图像处理、语音处理等方面都表现出了优异的性能，得到了广泛的认可。作为深度神经网络中的一个基础算法，有很多资料介绍了卷积实现原理，但是不少人在学习之后，还是对其及其梯度的计算过程细节不够清楚。在这里，我想分享几个自己做过的小试验来加深大家对卷积及其梯度计算过程的理解。

## 卷积计算过程

在卷积神经网络中，卷积计算过程可以通过下面的动图（来自[此处](https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/pub?slide=id.g1245051c73_0_2184)）来理解：

![conv2d](/attaches/2017/2017-04-15-understanding-gradients-of-conv2d-in-experiments/conv2d.gif)


以上是对一个3通道输入图像进行卷积操作的过程，卷积核是一个[4,4,3]的3维矩阵。可以看到，当我们要计算卷积结果的某一层时，我们使用同一个卷积核在输入图像上面从左到右从上到下（图像的长和宽）依次滑动，每滑动到一个位置，我们就用卷积核和图像的对应部分数值计算点积（对应点数值相乘，然后再全部相加，即4*4*3次乘法和加法操作）得到输出层的对应点的值，然后随着滑动的进行，我们就得到这一层的卷积结果。

下面我们来看看具体的数值计算结果，为了简单起见，我们考虑输入图像为灰度图像（即单通道）的场景（来自[此处](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#about-this-tutorial)）：

![numerical no padding no strides](/attaches/2017/2017-04-15-understanding-gradients-of-conv2d-in-experiments/numerical_no_padding_no_strides.gif)

此例中的卷积核为：

![numerical no padding no strides filter](/attaches/2017/2017-04-15-understanding-gradients-of-conv2d-in-experiments/numerical_no_padding_no_strides_filter.png)

到此相信大家都已经了解卷积如何计算出来的了。我们使用tensorflow来证实一下上面的计算。以下是在iPython里面的运行结果：

![conv2d expement](/attaches/2017/2017-04-15-understanding-gradients-of-conv2d-in-experiments/experiment_conv2d.png)

可以看到输出的结果即上面动图的结果。

卷积过程中，还有padding和stride的概念，相对容易理解，这里就不赘述了。

## 卷积的梯度计算

了解了卷积的计算过程，我们不禁会想，卷积计算还是挺复杂的，要自己动手编程实现也并非易事。而且，由于结果矩阵的每一层是共享同一个卷积核的，在反向传播过程中，卷积核又是如何被更新的呢（即梯度是多少）？相信这个问题会困扰不少非科班出身进入机器学习领域的同学们。下面就让我们一起结合试验和源代码来揭示这一过程吧。

观察上面的计算过程，事实上，卷积计算可以转化为矩阵乘法来实现的。具体如下：

1. 把每一个卷积核都reshape为一个行向量，多个卷积核就形成了一个矩阵
2. 从输入图像中提取patch（即每一次滑动时覆盖到的矩形框中的数据），然后将其reshape为一个列向量，每一滑动都有这样的一个列向量，这样就可以形成另一个矩阵
3. 将步骤1和2中得到的矩阵进行矩阵乘法，就得到最终的结果

以上面的单通道图像卷积计算为例，转换为矩阵乘法之后即计算如下乘法：

{% katex [displayMode] %}

\begin{pmatrix}
0 \\
1 \\
2 \\
2 \\
2 \\
0 \\
0 \\
1 \\
2
\end{pmatrix}^T * 
\begin{pmatrix}
3 &3 &2 &0 &0 &1 &3 &1 &2 \\
3 &2 &1 &0 &1 &3 &1 &2 &2 \\
2 &1 &0 &1 &3 &1 &2 &2 &3 \\
0 &0 &1 &3 &1 &2 &2 &0 &0 \\
0 &1 &3 &1 &2 &2 &0 &0 &2 \\
1 &3 &1 &2 &2 &3 &0 &2 &2 \\
3 &1 &2 &2 &0 &0 &2 &0 &0 \\
1 &2 &2 &0 &0 &2 &0 &0 &0 \\
2 &2 &3 &0 &2 &2 &0 &0 &1
\end{pmatrix} = 
\begin{pmatrix}
12 \\
12 \\
17 \\
10 \\
17 \\
19 \\
9 \\
6 \\
14
\end{pmatrix}

{% endkatex %}

tensorflow内部实现实际上就是如此，见[如下代码(摘录核心部分)](https://github.com/tensorflow/tensorflow/blob/master/third_party/eigen3/unsupported/Eigen/CXX11/src/NeuralNetworks/SpatialConvolutions.h#L768)：
```c++
DSizes<TensorIndex, 2> kernel_dims;
kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
kernel_dims[1] = kernelFilters;
kernel.reshape(kernel_dims).contract(
    input.extract_image_patches(kernelRows, kernelCols, stride, stride,
            in_stride, in_stride, padding_type)
        .reshape(pre_contract_dims),
    contract_dims).reshape(post_contract_dims)
```

了解到这一层，大家就应该知道了，卷积的计算实际上跟简单感知机的计算本质上是一致的。由此我们可以得出的结论是其梯度计算也是类似的。

我们先回顾一下感知机中的梯度计算，在iPython中进行如下试验：

![linear gradient experiment](/attaches/2017/2017-04-15-understanding-gradients-of-conv2d-in-experiments/experiment_linear_gradient.png)

与导数计算一致，可以看到c相对于a的梯度其实就是b矩阵的值。如果b的维度为[2, 2]，那么结果是多少呢？

![linear gradient experiment 1](/attaches/2017/2017-04-15-understanding-gradients-of-conv2d-in-experiments/experiment_linear_gradient_1.png)

可以看到此时a得到的梯度为叠加b矩阵对应位置的值。到这里，大家应该已经了解了，在进行反向传播时，卷积核的更新梯度实际上就是图片对应位置的值相加。当然这里没有考虑激活函数的影响，当有激活函数时，梯度会经过链式方式传导到卷积核上。

我们来验证一下，求上面的卷积核梯度：

![convolution gradient experiment](/attaches/2017/2017-04-15-understanding-gradients-of-conv2d-in-experiments/experiment_conv_gradient.png)

即：

```
filter_weights[0, 0, 0] = sum([3,3,2, 0,0,1, 3,1,2]) = 15
filter_weights[0, 1, 0] = sum([3,1,1, 0,1,3, 1,2,2]) = 15
```

## 反卷积（转置卷积）计算

卷积过程将图像映射到feature map，同时我们也会要用到将feature map映射到图像的问题。比如在autoencoder网络中，我们要将编码之后的数据反编码回来，还比如在GAN中我们会遇到图像生成的问题。

观察卷积的过程，我们实际上可以定义一个卷积的逆过程，由于最终卷积操作会转化为矩阵乘法，将原图像左乘一个filter_weights矩阵，那么能不能使用得到的feature map右乘一个filter_weights转置矩阵来实现将图片还原的过程呢？当然是可以的，这一过程大家通常将其称作反卷积，反卷积计算在论文[Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)中被提出。需要注意的是，这里的计算并不是卷积的逆过程，只是卷积过程的一个变形，具体的讲即是为了得到原图，在计算时将filter_weights矩阵转置了一下而已。

还是采用上述过程中的数据来做实验：

![deconvolution gradient experiment](/attaches/2017/2017-04-15-understanding-gradients-of-conv2d-in-experiments/experiment_deconv_gradient.png)

我们得到`filter_weights_deconv`的梯度了，但是为什么全部都是116呢？我们来考虑一下计算过程，实际上正向传播时，反卷积相当于进行了如下计算：

{% katex [displayMode] %}

\begin{pmatrix}
12 \\
12 \\
17 \\
10 \\
17 \\
19 \\
9 \\
6 \\
14
\end{pmatrix} * 
\begin{pmatrix}
0 &1 &2 &2 &2 &0 &0 &1 &2
\end{pmatrix} = 
\begin{pmatrix}
0 &12 &24 &24 &24 &0 &0 &12 &24 \\
0 &12 &24 &24 &24 &0 &0 &12 &24 \\
0 &17 &34 &34 &34 &0 &0 &17 &34 \\
0 &10 &20 &20 &20 &0 &0 &10 &20 \\
0 &17 &34 &34 &34 &0 &0 &17 &34 \\
0 &19 &38 &38 &38 &0 &0 &19 &38 \\
0 &9 &18 &18 &18 &0 &0 &9 &18 \\
0 &6 &12 &12 &12 &0 &0 &6 &12 \\
0 &14 &28 &28 &28 &0 &0 &14 &28
\end{pmatrix}

{% endkatex %}

这个计算的结果是一个[9, 9]的矩阵，结果矩阵进行patch的反转（对应位置的值相加）就得到原图了。

到此，相信大家都已经知道反卷积的计算细节了。

## 总结

本文介绍了卷积及反卷积的数学计算过程，同时结合试验进行相互验证，由此加深对卷积过程的理解。

## 参考资料

[theano卷积教程](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#about-this-tutorial)
[论文 -- Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
[tensorflow相关源代码](https://github.com/tensorflow/tensorflow/blob/master/third_party/eigen3/unsupported/Eigen/CXX11/src/NeuralNetworks/SpatialConvolutions.h#L768)
[知乎问题：如何理解深度学习中的deconvolution networks？](https://www.zhihu.com/question/43609045)














