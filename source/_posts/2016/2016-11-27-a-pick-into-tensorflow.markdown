---
layout: post
title: "Tensorflow一瞥"
date: 2016-11-27 15:19:15 +0800
comments: true
categories:
- tensorflow
- AI
- Machine Learning
---

今天跟大家分享一下时下非常流行的一个机器学习框架：TensorFlow。希望大家可以一瞥TensorFlow的易用性和强大功能。

TensorFlow目前在我司的技术雷达上面处于assess阶段。

## TensorFlow是什么

TensorFlow诞生于Google公司Google Brain项目。其前身是一个名为DistBelief的系统，DistBelief是Google内部使用非常广泛的一个机器学习系统。TensorFlow作为github上面的一个很火的开源项目，它的第一个提交是在2015年11月。到现在也不过刚好一年时间。

TensorFlow提供的API库可以用于编写富有表现力的程序。同时TensorFlow底层使用c++实现，其性能也是不错的。

TensorFlow在系统设计上使用一个有状态的数据流图来描述计算。使用TensorFlow时，需要先定义好计算图，以便TensorFlow可以在内部进行分布式的调度，然后一般会使用向计算图填充数据的形式进行迭代计算。

TensorFlow支持的系统非常广泛，从移动设备到桌面电脑再到大型分布式系统，从CPU到GPU，TensorFlow都提供了支持。

TensorFlow为了便于高效率的开发，同时也是顺应社区的技术潮流，提供的是Python的API。同时，也可以直接使用C++进行开发。目前还有Rust，Haskell的方言支持。

<!-- more -->

## 为什么要用TensorFlow

### 良好而活跃的社区

#### 丰富的入门教程

TensorFlow有很多Tutorial入门教程，大大降低了入门的门槛。官方的教程已经不错了，社区技术爱好者们还贡献了很多相关的教程。我个人用过的教程，可以列举如下：

- [官方教程](https://www.tensorflow.org/versions/r0.11/tutorials/index.html)
- 更有深度的[MOOC教程](https://github.com/pkmital/tensorflow_tutorials)
- [Google在Udacity上面做的教程](https://classroom.udacity.com/courses/ud730/)
- [Tensorflow and deep learning, without a PhD](https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/pub?slide=id.g140797b42d_0_60)

#### 大量现成的机器学习模型

Google内部使用TensorFlow实现了很多性能很好的机器学习模型（这里的性能指模型表现好，如分类错误率低），这些模型也都在[github](https://github.com/tensorflow/models)上面开源了出来。如为图片生成标题的模型，识别街道名称的模型等等。我们可以方便的阅读学习这些模型，同时也可以作为一个很好的起点，用于研究设计自己的模型。

#### 提供了更简单的机器学习的接口

TensorFlow同时提供了一个[简单的机器学习接口](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn)。几行代码即可完成模型训练和使用。

#### 丰富的周边工具

TensorFlow还提供了用于可视化参数的TensorBoard，大大方便了模型的调优工作。同时还提供了Caffe到TensorFlow模型转换的工具。

### TensorFlow已经在业界广泛使用。

众所周知的如Google自己，京东，Uber，DeepMind，SnapChat，Twitter等等，都在公司内部使用这个框架进行机器学习的研究。

### Google趋势表现抢眼

从Google趋势来看，TensorFlow也已成为当前非常流行的机器学习框架了。

![TensorFlow in Google Trend](/attaches/a-pick-into-tensorflow/tf-googletrend.png)


## 一个简单的例子：在TensorFlow中使用Logistic Regression来进行图片分类

下面用一个简单的例子演示一下TensorFlow的使用。这个例子中我们会对MNIST手写数字图片库进行分类。

MNIST数据集的是一个非常基础而简单的用于机器学习的数据集。下载好这个数据集之后，可以看到其包含的图片如下（一个数字对应一张图片）：

![MNIST Overview](/attaches/a-pick-into-tensorflow/mnist-overview.png)

我们将要使用的分类模型也是基础的Logistic Regression模型。

![Model Overview](/attaches/a-pick-into-tensorflow/model-overview.png)

这个模型用数学公司来描述就是如下这样：

![Math](/attaches/a-pick-into-tensorflow/math.png)

模型对应的核心代码TensorFlow代码就是：

```python

# Describe Graph

Y = tf.nn.softmax(tf.matmul(XX, W) + b)
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

predict = tf.argmax(tf.nn.softmax(tf.matmul(X, W) + b), 1)

# Train the model
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(10000):
    batch_X, batch_Y = mnist.train.next_batch(100)
    c = sess.run([cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})

# Predict
labels = sess.run([predict], feed_dict={...})

```


完整的代码可以参考这里的[tutorial](https://github.com/martin-gorner/tensorflow-mnist-tutorial.git)
















