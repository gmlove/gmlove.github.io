---
layout: post
title: "RNN和LSTM从理论到实践二：RNN和LSTM模型"
date: 2016-12-11 10:12:51 +0800
comments: true
categories: 
- tensorflow
- AI
- Machine Learning
- RNN
- LSTM
- Deep Learning
---

本文是上一篇文章『RNN和LSTM从理论到实践一：词向量』的续文。

上一章中，我们了解了词向量怎样训练，并跟随udacity上面的例子及问题动手实践了Skip Gram和CBOW模型训练算法。我们也顺带看了一下什么是语言模型，以及基础的n-gram模型是怎么样的。这次我们将要在前面的基础上，看看RNN和LSTM模型是什么样的，并将和大家一起动手去实现一个LSTM模型用于生成一个句子。

## 我们的问题

先来看我们的问题，然后让我们带着问题，来学习RNN和LSTM。这次我们要解决的问题是：如何生成一个看起来还不错的句子。

我们之前介绍过n-gram，那么我们能不能使用n-gram去预测单词，进而生成一个句子呢？我们可以使用频率统计来计算n-gram的语言模型：

<!-- more -->

![N-gram Equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/n-gram-equation.png)

如果我们要计算单词w1出现的条件下单词w2出现的概率，我们可以先统计单词w1在我们的训练数据集中一共出现了多少次，即count(w1)，然后再统计w1和w2连续出现，即"w1 w2"出现的次数count(w1, w2)，相除即为w1出现的条件下单词w2出现的概率。同理，我们想要求在w1和w2同时连续出现的情况下，w3出现的概率也可以类似地使用频率统计来求得。

n-gram看起来很容易实现，但是它能为我们生成一个好的句子吗？

试想，如果我们有一篇关于西班牙和法国的文章，里面有一句话『这两个国家开始进入战争时代』。那么这两个国家指的是哪两个国家呢？用n-gram模型其实很难回答这个问题，因为西班牙和法国可能是文章刚开始的时候指明的，而n-gram中的N不可能太大，太大通常导致内存不足以及计算太慢，所以n-gram无法获知文章中离得很远的句子的信息。

## RNN

现在我们知道n-gram的缺点了，那么如何能解决远距离信息问题呢？这就是我们要介绍的模型RNN的发挥价值的地方。

RNN的模型如下：

![RNN Model](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/rnn-model.png)

在这个模型图中，x表示输入的单词向量，y表示预测的单词向量，在此基础上，我们增加了h矩阵来表示输出特征。t表示时间。观察t时刻的输入单词x和预测单词y的计算过程，我们可以发现y不仅仅是跟输入x相关，还跟t-1时刻的h相关。而t时刻的h，又会贡献到t+1时刻的y预测过程中去。

具体的算法公式如下：

![Ht & Yt equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/ht-and-yt-equation.png)

我们可以看到我们定义了3个权值矩阵，这些权值矩阵在每次进行计算的时候，都是复用的，内存开销不随训练文本的增加而增加。h的计算使用了sigmoind激活函数，y的计算使用softmax激活函数。这里为了简单，我们省略了偏置向量。

好了，我们已经有了模型，那么我们的损失函数是什么呢？其实还是跟之前一致，在时刻t，我们使用交叉熵损失函数：

![Loss function of RNN](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/loss-function-of-rnn.png)

那么总的损失函数就是（T表示我们训练时只考虑T个单词形成的RNN）：

![Overall loss function of RNN](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/overall-loss-function-of-rnn.png)

我们还常常用perplexity来衡量模型的损失，perplexity的定义就是：

![Perplexity equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/perplexity-equation.png)

由指数函数的曲线可知，J越小时，perplexity也越小，而且perplexity始终为正。

### 反向传播的问题

直接使用RNN模型有没有什么问题呢？由于权值矩阵是共享的，在反向传播的时候，每一步都会更新一个相同的权值矩阵。梯度下降方法在这种情况下的表现是会有问题的，因为这里的更新会变得不稳定，很容易的就会导致更新太多或太少，从而产生梯度消失或者梯度爆炸的问题。

例如，假设有两个句子：

- "Jane walked into the room. John walked in too. Jane said hi to ___"
- "Jane walked into the room. John walked in too. It was late in the day, and everyone was walking home after a long day at work. Jane said hi to ___"

RNN更可能能将第一个句子预测正确，由于梯度下降的梯度消失问题，离得远的信息难以传播到当下，第二个句子将会更难预测准确。

事实上，可以计算得到，早于当前时刻的某时刻k的单词给当前时刻t的贡献将为β^(t-k)，可以看到它的梯度传播是指数递减的。这就很容易产生梯度爆炸和梯度消失问题。

针对这个问题，我们的解决方案就是：

- 在梯度将要爆炸的时候，将其裁剪为一个较小的值，这可以应对梯度爆炸问题
- 使用ReLU激活函数，更仔细的初始化权值矩阵，这可以解决梯度消失的问题

这些手段可以在一定程度上优化梯度爆炸和梯度消失的问题。但是其实我们有更好的模型来解决这个问题，那就是LSTM。我们后面会一起学习LSTM。

### RNN的扩展

通常在RNN上面还可以继续扩展，比如扩展为双向递归神经网络：

![Bidirectional RNN](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/bidirectional-rnn.png)

相关的公式就变为：

![Bidirectional RNN equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/bidirectional-rnn-equation.png)

我们还可以加深这个网络，让每一个时刻t从一个线性层变为多个线性层，如图：

![Deep Bidirectional RNN](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/deep-bidirectional-rnn.png)

相关的公式为：

![Deep Bidirectional RNN Equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/deep-bidirectional-rnn-equation.png)

## LSTM

前面我们分析了RNN的问题，并提到了LSTM。那么LSTM是什么东西呢？它是Long Short Term Memory的缩写。从这个全称来看，它可以给RNN引入记忆的功能。事实上他就是以这个为目标来进行设计的。LSTM的思想就是将RNN的网络单元替换为带记忆功能的网络单元。

![RNN to LSTM](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/rnn-to-lstm.png)

记忆到底是个什么玩意儿呢？听起来似乎很抽象。我们分析一下记忆单元应该有的功能。与人的记忆能力作为对比，一个单元能有记忆，那么它应该可以被写入，即可以被更新，然后应该可以被读取，同时应该可以选择性的忘记，也就是删除数据。

![LSTM functionality](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/lstm-functionality.png)

### Gated Recurrent Units

分析了LSTM的原理，我们先看看GRU，即Gated Recurrent Units，它是基于这个思想的一个比LSTM更简单的模型。它的网络图如下：

![Gated Recurrent Units](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/gru.png)

Reset其实用于实现忘记功能的，它可以控制是否将上一时刻的输出特征h即此时刻的输入包含到此时刻的预测过程，或者包含多少，其公式如下：

![GRU Reset Equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/gru-reset-equation.png)

New Memory可以实现写入控制的功能，即决定现在的输入和上一时刻的h到底有多少会影响到当前的预测过程，其公式如下：

![GRU New Memory Equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/gru-new-memory-equation.png)

Update可以实现控制读取的功能，即决定当前的状态有多少可以被下一时刻读取，其公式如下：

![GRU Update Equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/gru-update-equation.png)

最后将以上几个控制单元结合起来就得到当前的特征输出，它也将会被用于预测输出，其公式如下：

![GRU feature equation](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/gru-feature-equation.png)

可以看到这些单个功能的控制器几乎都是一样的，他们的功能是在构造输出特征时体现出来的。

### LSTM Units

我们再看看更复杂一些的LSTM模型，其模型图如下：

![LSTM](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/lstm.png)

公式如下：

![LSTM equations](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/lstm-equations.png)

其中h用于预测，c和h一起传播到下一时刻。

### LSTM的应用

通过分析RNN和LSTM的特性，可以发现它们可以用于预测一个序列。当训练好了LSTM网络时，得到一个输入之后，就预测下一个输出，然后结合输入和预测到的输出，可以继续预测下下个输出，连续的预测之后，我们就可以得到一个序列了。在我们后面要训练的模型中我们就是采取这样的方式来生成一个序列的。就如同下面这样：

![LSTM Application](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/lstm-application.png)

当然我们还可以在每次预测时选择top k个可能的结果，每次都这样选择，然后在预测过一定次数n之后，统计生成的n个词的短序列的概率，并选择概率最大的短序列作为最终的结果。这样可以防止在某一步预测失败之后，导致后面的预测都跟着失败。这个想法就是Beam Search算法的思想。

![LSTM Beam Search](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/lstm-beam-search.png)

RNN和LSTM还有很多其他方面的应用，由于RNN和LSTM可以生成一个连续的序列，我们可以将其应用于机器翻译、语音识别、以及根据图片生成标题等等。

## 训练一个RNN和LSTM的模型

下面我们将跟随udacity上面的习题，一起来实践这个算法。

下面的内容，请结合[这里的代码](https://github.com/gmlove/dl-workshop-rnn-lstm/blob/master/6_lstm.ipynb)来阅读。

### 示例代码中的模型

在示例中，为了简化模型，我们这里对字符进行建模。于是我们的字符就可以用大小为27（a-z和空格）的一个one-hot的向量来表示。我们的RNN模型中，将按照10个字符进行分组递归处理，并使用变量num_unrollings来表示10这个数值。于是训练数据将被分为每10个字符一组，这一组字符进入RNN之后，不仅相互连接起来，而且和前一组的输出连接，同时提供输出作为后一组的输入。这样分组更新权值的过程中，每次迭代就会将梯度传播10次。

下面我们看看具体实现过程。

首先还是通过函数`maybe_download`和`read_data`来下载和读取数据，然后选择1000个字符的数据作为验证集。

然后，我们准备两个工具函数，将字符映射到索引，和将索引映射到字符。

```python
vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '
```

我们现在来考虑如何进行batch迭代。按照我们前面的分析，由递归的特性可知，字符需要首尾相连，训练集中的后一个字符是依赖前一个字符进行计算的，我们无法打乱这个顺序。我们这里的想法就是把训练文本分为n份，然后n份一起开始训练。如下图所示：

![Batch update](/attaches/2016/2016-12-11-dl-workshop-rnn-and-lstm-1/batch-update.png)

理解了这个之后，代码中的`BatchGenerator`就比较容易理解了。需要注意的是batch中的字符组长度为11，并且一个batch和下一个batch之间是有一个字符重叠的，这是由于我们会使用每一个字符的后一个字符作为标签字符，所以在生成batch的时候，就故意多生成了一个字符。在编写BatchGenerator的同时，我们编写了两个辅助函数`characters`和`batches2string`，分别实现将一个概率分布（即softmax的输出）映射到一个字符，和将一组batch映射为字符串。

```python
batch_size=64
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))
```


然后为了程序需要我们再准备几个工具函数如下：

```python
def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]
```

其中`logprob`用于计算交叉熵，`sample_distribution`用于从一个概率分别里面随机选择一个元素，`sample`在一个概率分布中随机选择一个元素，并将其转换为one-hot编码的向量。`random_distribution`可以随机生成一个概率分布。

现在到了我们构建计算图的时候了，参考lstm的模型图，我们依序初始化参数并构造计算图如下：

```python
num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
  cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
```

`tf.control_dependencies`这个是用于控制执行顺序的，我们想要完成一组字符的首尾连接的计算图之后，再进行loss函数的构建。`saved_output.assign(output)`和`saved_state.assign(state)`可以将10个递归的lstm串起来，`tf.control_dependencies`保证了loss肯定在assign之后构建。这样就可以保证我们的计算图是全部递归的结构构建的。

接下来是训练的代码：

```python
num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size)))
```

迭代训练过程中，将会每100个迭代计算一次训练数据的perplexity，然后打印出来。并且在1000个迭代的时候，尝试随机选择验证集中的数据进行预测，并与真实的数据进行对比计算perplexity。

### 问题1

现在来看我们的问题1。观察计算图构建过程，看起来是可以优化的，因为各个控制器的代码看起来很类似。试想，如果我们将所有的`tf.matmul(o, im) + ib`都抽取出来组合为一个大的矩阵之后再做乘积，然后利用矩阵乘法的规则，将乘积之后的结果分配到各个控制器的结果中是不是可以呢？下面我们就用这个思想来优化我们的模型。

下面的内容，请结合[这里的代码](https://github.com/gmlove/dl-workshop-rnn-lstm/blob/master/6_lstm-problem1.ipynb)来阅读。

我们只需要将权值定义代码替换为：

```python
  # Parameters:
  # Input gate: input, previous output, and bias.
  ifcox = tf.Variable(tf.truncated_normal([vocabulary_size, 4 * num_nodes], -0.1, 0.1))
  ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
  ifcob = tf.Variable(tf.zeros([1, 4 * num_nodes]))
```

并将`lstm_cell`函数替换为：

```python
def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    all_gates = tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob
    input_gate = tf.sigmoid(all_gates[:, 0:num_nodes])
    forget_gate = tf.sigmoid(all_gates[:, num_nodes:2*num_nodes])
    update = all_gates[:, 2*num_nodes:3*num_nodes]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates[:, 3*num_nodes:])

    return output_gate * tf.tanh(state), state
```

### 问题2-1

我们在这里会尝试编写一个bi-gram的实现。我们之前是对字符进行建模的，转到bi-gram上，也就是说我们需要在一对一对的字符上进行建模。

下面的内容，请结合[这里的代码](https://github.com/gmlove/dl-workshop-rnn-lstm/blob/master/6_lstm-problem2-1.ipynb)来阅读。

首先是改写我们的工具函数（此时我们的词库大小为27*27）：

```python
vocabulary_size_base = len(string.ascii_lowercase) + 1 # [a-z] + ' '
vocabulary_size = vocabulary_size_base ** 2

first_letter = ord(string.ascii_lowercase[0])
def char2id0(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0

def char2id(char):
  return char2id0(char[0]) * vocabulary_size_base + char2id0(char[1])
  
def id2char0(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

def id2char(dictid):
  return id2char0(dictid//vocabulary_size_base) + id2char0(dictid%vocabulary_size_base)
```

然后我们还需要修改`BatchGenerator`来生成batch的训练数据。我们还增加了一个`bigramstringtonormal`函数来将bi-gram编码的字符串转换为一个可读的字符串。

之后在训练过程中，我们需要打印预测出来的字符串，这时需要调用我们的工具函数将其转换为可读的字符串再打印出来。

需要修改的代码请参考[这里的代码](https://github.com/gmlove/dl-workshop-rnn-lstm/blob/master/6_lstm-problem2-1.ipynb)。

### 问题2-2

我们将会引入embedding来节省内存。

需要修改的代码请参考[这里的代码](https://github.com/gmlove/dl-workshop-rnn-lstm/blob/master/6_lstm-problem2-2.ipynb)。

主要的思路是，生成batch训练数据时，无须进行one-hot编码了，直接生成一个二维表即可。`ifcox`权值矩阵在这里充当了embedding的角色。然后在构造lstm单元的时候从embedding查询得出乘积后的值，而无须再进行乘积处理。在最后计算loss的时候，将`softmax_cross_entropy_with_logits`转换为调用它的稀疏矩阵版本`sparse_softmax_cross_entropy_with_logits`。

### 问题2-3

我们将会引入dropout来提升性能。

需要修改的代码请参考[这里的代码](https://github.com/gmlove/dl-workshop-rnn-lstm/blob/master/6_lstm-problem2-3.ipynb)。

需要注意的是向lstm单元添加dropout的时候，只能加到input数据的处理上面，而不能加到递归连接的处理单元上面。

### 问题3

暂无

## 总结

到这里我们就完成了RNN和LSTM的学习了。结合NN CNN可以看到，深度学习的基础其实很简单，就是线性模型加激活函数，常用的深度学习模型其实就是在这样的基础模型上面进行结构的设计和优化。这跟玩乐高类似，由简单的基础的模块进行组合，就可以得到非常复杂的模型，最后的效果是惊人的。

我们接触过的基础的模型有：

线性单元： y=WX+b
激活函数： sigoid tanh relu softmax(一般用作输出层)
优化手段：normalization, randomization, l2 regularization, dropout, embedding, gradient clipping










