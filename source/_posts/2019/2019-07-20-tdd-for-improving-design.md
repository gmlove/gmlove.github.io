---
title: 从改善设计的角度理解TDD
categories:
- tdd
- 敏捷
tags:
- tdd
- agile
- 敏捷
- 测试
- 质量
date: 2019-07-20 20:00:00
---

TDD有很多好处，但是广大程序员却总是难以接受。即便在我们ThoughtWorks，有着非常浓厚的TDD氛围的公司里，接受起来也依然不是一件简单的事情。我曾经见过一些在我们公司工作过一年甚至两年的同事，对TDD的理解都还停留在比较粗浅的认识上，平时的实践也难以跟上。

<!-- more -->

为什么我们在接受这样一个优秀的实践的时候会这么困难呢？我认为是没有真正体会到TDD给我们带来的好处。我们通常说TDD，一般都是给大家强调软件质量，有了TDD，就有测试，软件质量就有保障了。这是最直接的一个好处，但是由于对于软件质量理解的不同，不少人在心底里并不是完全认同这一点的。测试其实只能说是TDD带来的好处之一，长时间实践下来，我觉得TDD带来的更多的是软件设计能力的提升。相信广大程序员普遍认同设计能力的高低是区分优秀程序员的一个重要标准，从提升设计的角度理解TDD，相信我们就更容易接受了。

TDD是如何帮助提升设计能力的？下面我将分享TDD在我们项目日常代码实践中的应用，希望能帮助提升对TDD的理解。

项目中有这样一个例子，我们的机器学习模型在训练开始的时候需要加载数十G的数据到内存（可用内存很大），这个数据加载过程比较慢，这让调试模型参数的需要等待较长时间才能看到结果。一个自然的想法就是先加载一部分的数据到内存，然后启动训练，再启动一个后台任务去加载其他的数据。（在这里，数据是一系列的大文件，约50个，每个文件600M大小。）如何实现这个功能呢？

这个任务的需求是比较明确的，我们从测试的角度出发考虑来看一下如何进行程序设计。为了编写测试，第一个问题是如何给测试命名。从需求来看，这里应该是完成数据加载的功能，结合面向对象的思想，好，那么我们就叫它数据加载器吧，`DataLoader`，听起来不错。于是就可以写下第一行测试代码了。如下：

```python
import unittest

class DataLoaderTest(unittest.TestCase):
    ...
```

这里的功能是要完成数据加载，那么接着写下测试用例的名字。

```python
import unittest

class DataLoaderTest(unittest.TestCase):
    def test_should_load_data_correctly():
        ....
```

测试代码应该是什么样的呢？DataLoader这个对象其实是需要对机器学习模型提供数据支持的，如何提供数据支持需要结合后续模型如何使用来确定。


由于我们想要做到异步数据加载，看了keras的[文档](https://keras.io/models/sequential/#fit_generator)就会知道需要使用`fit_generator`这种方式来实现，而这个函数要求传入一个python的`generator`或者一个`keras.utils.Sequence`对象。好了，那么我们就需要使用这两个接口兼容的方式来设计接口。

为实现`keras.utils.Sequence`，需要实现接口`__len__`和`__getitem__`来获取总的batch的数量和每一个batch的数据。`generator`的方式似乎更简单，但是`fit_generator`需要我们传入迭代的次数`steps_per_epoch`，所以同样需要正确的知道总的数据量的大小。好了，我们需要实现的接口到这里就大致确定了。

从前面的分析来看，如果只是有一个数据加载器似乎不够，因为我们是需要给后续的训练提供整个数据访问支持的。OK，我们可以稍微扩展一下它的功能，不如叫`Dataset`吧，由于是异步加载数据的，可以叫`AsyncLoadedDataset`。按照这样的设计，修改测试如下：

```python
import unittest
import time

class AsyncLoadedDatasetTest(unittest.TestCase):

  def test_should_be_able_to_get_data_from_dataset_correctly():
    dataset_path, batch_size = '/tmp/data', 5
    dataset = AsyncLoadedDataset(dataset_path, batch_size)

    self.assertEqual(20, dataset.batch_count())
    self.assertEqual(pickle.load(open('/tmp/data/X.0.pickle', 'rb'))[:5],
        dataset.get_batch(0))
    batch_idx_for_async_loaded_data = 10
    self.assertEqual([], dataset.get_batch(batch_idx_for_async_loaded_data))

    time.sleep(5)

    self.assertEqual(pickle.load(open('/tmp/data/X.1.pickle', 'rb'))[:5], 
        dataset.get_batch(batch_idx_for_async_loaded_data))
```

这里，通过测试看到了我们的类应该要提供什么样的接口，以及应该如何工作。首先是它的构造，我们传入了一个文件路径和一个`batch_size`，因为我们需要指定它从什么地方加载数据，以便可以加载准备好的测试数据，同时指定批大小之后，后续测试中就可以根据这个大小计算出相应的期望的数据。其次是接口的名字，根据需求来，使用业务术语，我们将其命名为`batch_count`和`get_batch`。（有人可能会问名字怎么得来的，这里的函数名的其实是来的非常自然的。读一下`dataset.batch_count`，就可以知道其指batch_count of this dataset，也就是这个数据集的批数量。读一下`dataset.get_batch`，就可以知道是从数据集获取某一批的数据。也就是说这里的名字只要取得让我们读起代码来符合英语阅读习惯就好了。当然如果用`get_batch_count` `get_batch_data`在命名一致性上更好，是不是也可以呢？当然也是可以的，这些小的变化过于细节，不用太纠结，最终以读代码是否像读文章一样通畅为标准就行。）

到这里我们就将我们想要的类的设计完成了。而且经过我们在测试中的使用，这个类应该是易于使用的。我们对于这个类的设计应该是比较满意的。这个类的定义几乎是呼之欲出，如下：

```python
class AsyncLoadedDataset:

  def __init__(self, data_dir: str, batch_size: int):
    pass

  def batch_count() -> int:
    pass

  def get_batch(batch_idx: int) -> Union[List, np.ndarray]:
    pass
```

我们来回顾一下上面的过程，仔细体会一下TDD是如何帮助完成一个让我们满意的设计的。有以下几点，可以归功于TDD：

1. 从测试开始写，这帮助我们梳理清楚了需求。让我们从开始理解的`DataLoader`推进到后续理解的`AsyncLoadedDataset`，我们加深了对需要解决的问题的理解
2. 从测试开始写，这帮助我们从面向对象的角度进行程序设计，抽象了一个`AsyncLoadedDataset`对象
3. 从测试开始写，这帮助我们设计了正确且易于使用的类的构造器
4. 从测试开始写，这帮助我们从代码阅读者的视角出发给函数命名，从而得到一个更好的名字。

上述“帮助”一词，其实可以完全替换为“驱动”，这也就是测试驱动开发了。

（有人会争辩这个测试是不是好的单元测试，因为这个测试事实上是对外部的数据产生了依赖，而且测试中有`time.sleep`出现，运行会很慢。这很正确，我们这里的测试本来就不是一个单元测试，它的定位应该是一个小的集成测试。为什么要用这个小的集成测试呢？我们这里做的其实是实现功能的第一步，第一步当然是站在功能完整的角度来看问题。并且，我们需要测试的是线程能不能正常工作，数据能不能正常加载，并正确计算。这里问题的核心就涉及到和线程模块以及数据加载模块的集成。所以我们的测试就写成了一个小的集成测试。

为了让这个测试变得更易于维护，我们可以准备一份很小的测试数据放到我们的测试资源里面当做代码的一部分管理起来。我们甚至还可以mock掉线程的API，让我们的测试运行更快，但是这样做了之后，这个测试是不是会降低我们对代码正确的信心呢？在这里，易于维护和增强信心有一定的冲突，我也没有答案，这可能是大家要去平衡的一个问题。）

到这里我们可以开始写我们的代码了。我们可以编写这样的实现：

```python
class AsyncLoadedDataset:

  def __init__(self, data_dir: str, batch_size: int):
    self.data_dir = data_dir
    self.files = os.listdir(data_dir)
    self.files.sort()

    self.batch_size = batch_size

    self.x_train, self.y_train = [], []
    self._load_data(self.files[0:1])

    self._loader_thread = Thread(target=self._load_data, args=self.files[1:], deamon=True)
    self._loader_thread.start()

  def _load_data(self, files: List[str]):
    for f in files:
      x, y = pickle.load(open(os.path.join(self.data_dir, f), 'rb'))
      self.x_train.append(x)
      self.y_train.append(y)

  def batch_count() -> int:
    pass

  def get_batch(batch_idx: int) -> Union[List, np.ndarray]:
    pass

```

(这里的实现没有考虑一些边界情况，比如当数据文件数据数量少于2个时，因为我们实际用的时候没有这些问题，为了快速实现，先忽略这些问题。)

数据加载这一部分可以很容易的实现，但是后续关于batch的处理要怎么实现呢？仔细想一下，这里的batch处理还是有点复杂的。第一，我们要拿到所有的数据量大小才能计算总的batch数量，如果全部读一遍肯定很慢，让我们的异步加载失去价值；第二，读取任意一个batch的数据时候，有时候可能会跨文件访问数据。

对于第一个问题，我们可以设计一个缓存，在使用数据集之前构建好这个缓存，比如我们可以规定数据目录里面必须有一个`total_count`的文件来指定总共的数据量大小。

第二个问题有一个非常简单的实现方式，那就是将所有的已加载数据组成一个新的list，然后整体上去计算batch对应的数据索引。但这里我们的数据量很大，我们需要避免数据拷贝产生大量的内存消耗，还需要避免连接列表带来的性能开销。这里我们可能需要根据每个文件中数据量的大小去计算索引，这个问题在逻辑上就比较复杂了，我们很难有信心一次性写对。但这是一个不错的方案。这个时候，我们就想，能不能对这一块的代码建立一个测试呢？这种逻辑复杂的情况下，测试可以辅助我们更容易的写出正确的代码。要建立这样一个测试，我们希望最好能独立于之前的测试存在，因为这样的测试会更容易写。为了达到这一目的，进一步思考，我们是不是可以将这个问题抽象成为一个更通用的问题？这里我们本质上是想以和单个列表访问数据类似的方式从多个列表中访问数据。好了，我们可以抽象一个通用的类`MultiList`来表达这个想法。`MultiList`就是一个由多个列表组成的，表面看起来像一个列表的东西。

有了这里的分析，我们可以得到我们的测试如下：

```python
class MultiListTest(unittest.TestCase):

  def test_should_get_data_from_multi_list_as_the_provided_index_range():
    ml = MultiList(list(range(10)), list(range(10, 20)), list(range(20, 30)))
    self.assertEqual(list(range(0, 5)), ml.get_range(0, 5))
    self.assertEqual(list(range(5, 8)), ml.get_range(5, 8))
    self.assertEqual(list(range(5, 15)), ml.get_range(5, 15))
    self.assertEqual(list(range(5, 21)), ml.get_range(5, 21))
    self.assertEqual(list(range(5, 30)), ml.get_range(5, 100))
```

有了这个测试，我们的实现还会远吗？有兴趣的小伙伴可以当做练习完成后面对于这个`MultiList`及`AsyncLoadedDataset`的实现。这里主要是帮助我们体会完成设计的过程，从而体会TDD给我们带来的好处，对于后续细节就不赘述了。我们再来回顾一下TDD是如何帮我们完成设计的：

1. 从测试的角度出发，我们做了更进一步的抽象，从而得到了一个通用的`MultiList`对象
2. 从测试的角度出发，我们完成了对象的构造器及方法的设计
3. 从测试的角度出发，我们完成了对于对象的功能的定义，从而也展示了对象的使用方法

通过上面这个例子，相信大家能感受到TDD给我们设计带来的好处。总结起来，TDD可以辅助提升面向对象设计水平，TDD可以辅助提升代码可读性，
TDD可以辅助理解并应用SOLID原则进行程序设计。一句话总结TDD在实践中发挥的作用，我认为是，因为TDD让我们从使用者的角度去看待我们的设计，为了方便我们自己的阅读和理解，我们会自然的得到易于使用的设计，从而自然的就让我们的设计变得更好了。

通过上面的经验的分享，不知道大家是不是更认可和接受TDD了呢？但是要熟练运用起来，关键还是在于刻意的去练习。这里面要写好测试技巧其实就是比较多的，不过每天的日常工作都是机会，希望大家能保持开放的心态，严格要求自己，遇到问题多讨论交流。当团队中所有人的代码能力都上去了的时候，我们才能说我们是一个高效的团队，我们能做高质量的产品。所以，加油吧！

