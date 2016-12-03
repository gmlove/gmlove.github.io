---
layout: post
title: "RNN和LSTM从理论到实践一：词向量"
date: 2016-12-02 23:22:40 +0800
comments: true
categories: 
- tensorflow
- AI
- Machine Learning
- RNN
- LSTM
- Deep Learning
---

本文试图帮大家理解深度学习中的两大重要而基础的模型RNN和LSTM，并结合google在udacity上面关于深度学习的课程习题进行实践。

近两年深度学习在自然语言处理领域取得了非常好的效果。深度学习模型可以直接进行端到端的训练，而无须进行传统的特征工程过程。在自然语言处理方面，主要的深度学习模型是RNN，以及在RNN之上扩展出来的LSTM。RNN和LSTM也可以广泛用于其他序列处理和预测的机器学习任务。

RNN，全称为Recurrent Neural Network，常译为循环神经网络，也可译为时序递归神经网络，很多人直接简称为递归神经网络。另一个模型Recursive Neural Network，缩写也同样是RNN，译为递归神经网络。递归神经网络是时序递归神经网络的超集，它还可以包括在结构上有递归的神经网络，但是结构递归神经网络使用远没有时序递归神经网络使用得广泛。

本文包括四个部分：

- NLP
- 单词的向量表示
- RNN和LSTM理论介绍
- 训练一个LSTM模型

<!-- more -->

## NLP

我们首先来看看自然语言处理。自然语言处理可以说是信息时代最重要的技术之一，实际上自然语言处理无处不在，因为人们几乎所有交流沟通都通过语言进行，如搜索，广告语，邮件，翻译等等。

我们可以列举自然语言处理中的部分任务如下：

- 简单的任务：拼写检查、关键词搜索、同义词查找
- 比较复杂的任务：从网站或者文档中提取信息
- 很困难的任务：机器翻译、语义分析（在用户使用搜索引擎时，他输入的查询是什么意思？）、指代分析（如：文档里面的『他』或『她』具体指谁？）

## 单词的向量表示

要解决自然语言处理中的问题，我们首先要解决的问题是，如何表示这些问题。回顾之前的课程，我们可以发现，通常我们都会将输入数据表示成向量，在向量上进行数学建模。对于自然语言处理，也是一样的，我们要想办法将输入数据转化为向量。这里我们只讨论英文语言处理。那么究竟应该怎样用向量表示英文中的单词呢？

我们最容易想到的方法就是，跟之前的课程中对类别的处理一样，直接做one-hot编码。将所有单词排序，排序之后每个单词就会有一个位置，然后用一个与单词数量等长的数组表示某单词，该单词所在的位置数组值就为1，而其他所有位置值都为0.

![One-hot encoding for words](/attaches/dl-workshop-rnn-and-lstm/one-hot-encoding-for-words.png)

但是这样做有什么问题呢？第一个问题就是这样编码太稀疏了，会导致维度非常高，因为单词的数量级通常在10^6级别，维度高就导致计算困难。第二个问题是我们无法简单的从这样的编码中得知单词之间的关系。

为什么单词之间的关系重要呢？因为在我们使用语言时，单词之间并非完全相互独立的。比如短语"at work"，"at"和"work"之间存在一种可搭配使用的关系。而我们要进行语言分析时，单词之间的关系使用就更频繁了。我们来看看一个单词间关系的例子。

下面的习题答案是什么呢？

![Word analogies question](/attaches/dl-workshop-rnn-and-lstm/word-analogies-question.png)

"puppy"对"dog"增加了宠物的属性，那么"cat"加上宠物属性就变成了"kitten"。

"taller"对"tall"增加了比较级属性，那么"short"加上比较级属性就变成了"shorter"。

那么问题来了，如何进行机器学习训练才能得到这样的关系属性呢？先看两个句子。

![Way to find relationship between words](/attaches/dl-workshop-rnn-and-lstm/way-to-find-relationship-between-words.png)

如果说在我们的训练数据中出现了四个句子：

- The cat purrs.
- This cat hunts mice.
- The kitty purrs.
- This kitty hunts mice.

那么我们就有了一个很强的推断：*cat和kitty是相似的*。

**我们用于提取这种关系的方式就是：**

- 使用低维向量来表示单词
- 用邻近的单词来进行相互预测

### 语言模型

在进行实践之前，我们先来看看一个重要的背景知识：语言模型。

早期的自然语言处理采用硬编码的规则来实现。在上世纪80年代，机器学习被应用于自然语言处理中，统计语言模型被提出来，并广泛应用于机器学习模型中。我们这里的语言模型就是指统计语言模型。

我们认识一下什么是一个好的模型？对某个我们认为正确的句子，比如『狗啃骨头』，一个好的模型将能给出很高的概率。而对于不合理的句子，比如『骨头啃狗』它将给出很低的概率。这里面的一个重要的概念就是句子的概率。统计语言模型，简单而言，就是计算某一个句子的概率：P(w1, w2, w3, ...)。其中w表示句子中的单词。

如何计算这样的概率呢？为了简便处理，我们可以根据前n个词来预测下一个词。这样我们就得到了Unigram Model，Bigram Model, Trigram Model或者N-gram Model。

Unigram Model是指，我们可以将每个单词视为独立无关的，于是可以得到下面的等式：

![Unigram model](/attaches/dl-workshop-rnn-and-lstm/unigram-model-e1.png)

Bigram Model是指，如果当前单词只依赖其前面一个单词，在『狗啃骨头』中就表示可以用『狗』来预测『啃』。这样的话，我们的模型就可以用下式计算（P(w2|w1）表示在出现单词w1时，出现w2的概率)：

![Bigram model](/attaches/dl-workshop-rnn-and-lstm/bigram-model-e1.png) 

Trigram和N-gram Model可以得到的等式如下：

![Trigram and N-gram Model](/attaches/dl-workshop-rnn-and-lstm/trigram-ngram-model-e1.png)

事实上直接使用N-gram模型来计算句子概率是有问题的。因为它太简单了，最多能表示单词和前n个单词的关系，前n+1个单词就无法表示。而且n不能太大，太大会导致计算问题，并且n太大通常性能不会有明显的提升。

### Word2vec

回到词向量这个主题，对于词向量模型，我们要介绍的是word2vec算法。Word2vec从这个名字简单易懂，但是它似乎概括了所有提取词向量的算法，从这个名字我们大家可以想象一下它在自然语言处理中的地位。该算法是google于2013年提出来的，一经提出便被广泛应用了起来。

word2vec算法，在不断发展沉淀之后，得到两个机器学习模型：Skip Gram Model和CBOW(Continuous Bag of Words)。Skip Gram Model在实现上相对简单，而且google在udacity上面的题目也是以Skip Gram Model作为引子。我们先看看Skip Gram Model，然后在后面的习题中再一起看看CBOW模型。

### Skip Gram Model

Skip Gram Model属于非监督学习领域，这跟之前的图片识别不同。图片识别时，对于每一张图片我们是有标签的，比如某一张内容为"A"的图片，那么它的标签就是"a"。对于文本而言，原始数据只有一堆文本，一长串的单词序列。我们是没有显示的给定任何标签的。但是机器学习算法又是需要标签的，要不然我们无法计算我们的损失函数。对于这个问题，我们的想法是通过文本内容构造标签。借鉴N-gram模型的想法，如果单词只跟周边的单词相关，那么我们是不是就可以说在使用单词进行预测时，周边的单词就是该单词的正确预测结果呢？Skip Gram Model就是基于这个想法。

![Skip gram model](/attaches/dl-workshop-rnn-and-lstm/skip-gram-model.png)

这个算法的步骤如下：

- 随机生成一个大小为(vocabulary_size, embedding_size)的embedding矩阵（即所有单词的词向量矩阵，每一个行对应一个单词的向量）
- 对于某一个单词，从embedding矩阵中提取单词向量
- 在该单词向量上使用logistic regression进行训练，softmax作为激活函数
- 期望logistic regression得到的概率向量可以与真实的概率向量（即周边词的one-hot编码向量）相匹配

### Skip Gram Model中的问题及Negative Sampling

上面的算法最后一步在计算上是有问题的。根据公式 y = softmax(XW+b)，其中

- X的维度是：(batch_size, embedding_size)
- W的维度是：(embedding_size, vocabulary_size)

softmax将需要在(batch_size, vocabulary_size)矩阵上面进行计算，vocabulary_size通常是很大的，softmax在进行e^x运算时就会遇到计算问题。

我们应对这个问题的方法是Negative sampling。Negative sampling是指，我们在计算最终的softmax时，可以只选取部分错误的label和正确的label进行组合，而无须选择所有的错误label进行计算。当训练的迭代次数足够大时，这对于整个结果是没有影响的。在后面的作业中，num_sampled就是指选择的错误的label的数量。

### 使用word2vec之后如何衡量单词间的相似性？

在得到词向量之后，如何衡量单词间的相似性呢？

回顾一下我们的训练过程。如果两个单词相似，即出现了两个句子"The cat purrs"和"The kitty purrs"，那么cat的词向量经过计算之后可以得到purrs词向量，kitty词向量经过计算之后也可以得到purrs词向量。再结合softmax的比例计算过程，可以得出的结论是，最终的词向量里面，相似的单词，他们的词向量值在比例上也是相似的。

相似性就是：给定单词w1 w2 w3的词向量Vw1 Vw2 Vw3，如果Vw1 * Vw2.T > Vw1 * Vw3.T，那么我们就认为w2比w3更接近w1。事实上我们通常会用余弦距离去衡量词向量的相似性，即词向量间的夹角。

### Coding时间

[这个工程](git@github.com:gmlove/dl-workshop-rnn-lstm.git)里面包含了所有用到的数据及代码。下载工程之后，按照之前的办法，将整个目录映射到docker镜像中。

下面的内容，请结合[代码](https://github.com/gmlove/dl-workshop-rnn-lstm/blob/master/5_word2vec.ipynb)来进行阅读。

首先是下载、验证和读取文本。在代码中对应`maybe_download`和`read_data`。

```python
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
  
words = read_data(filename)
print('Data size %d' % len(words))
```

接下来是根据文本构造我们要用的数据结构。我们先定义我们的单词库大小，这里设置为50000。在调用`build_dataset`之后，得到的数据为：

- `count`：一个单词和它出现的次数的list，按照单词出现次数排序。如果单词出现次数太低，排序在50000之后，那么我们就将它映射到"UNK"单词，即unknown。这里的单词索引将用于把文本映射为整数值。
- `data`：原始文本映射到索引之后的序列。如原始文本为"anarchism originated as a term of abuse first"，映射之后为`[5243, 3083, 12, 6, 195, 2, 3136, 46, 59, 156]`
- `dictionary`：用于查询词对应的索引的字典
- `reverse_dictionary`：用于查询索引对应的单词的字典

```python
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.
```

然后我们需要有一个过程来生成batch，每次都批量处理数据，我们才能发挥计算机的并行计算的实力。

在生成batch时，我们期望在输入参数为`batch_size` `num_skips` `skip_window`时，可以对一小段文本，即skip_window长的文本，使用中心词来预测周边的词，生成num_skips个类似(words[2] -> words[0])这样的预测组。

当原始数据为`['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']`时，以num_skips=2和skip_window=1来调用generate_batch之后，得到的batch为`['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term']`，对应的label为`['as', 'anarchism', 'a', 'originated', 'term', 'as', 'a', 'of']`

```python
data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
```

有了这些数据结构和工具函数之后，我们就可以构造我们的模型如代码中所示。

- `embedding_size`：我们期望编码之后的词向量长度
- `train_dataset`：大小为batch_size的int数组，我们将单词索引在embedding矩阵中查找词向量
- `train_labels`：大小为[batch_size, 1]的矩阵，因为我们最终进行损失计算时，是使用one-hot编码的cross entropy
- `valid_examples`：从前100个词的单词集中随机选取16个单词用于计算相似度
- `valid_dataset`：对应于valid_examples，大小为16的int数组
- `embeddings`：大小为[vocabulary_size, embedding_size]的浮点型矩阵
- `softmax_weights`：大小为[vocabulary_size, embedding_size]的浮点型权值矩阵
- `softmax_biases`：大小为[vocabulary_size]的浮点型偏置向量

我们的模型就是先调用`tf.nn.embedding_lookup`去查找词向量。然后调用`tf.reduce_mean`和`tf.nn.sampled_softmax_loss`计算损失值。之后使用`tf.train.AdagradOptimizer`进行梯度下降，期望最小化损失值。梯度下降会同时优化我们的embeddings矩阵、softmax_weights矩阵及softmax_biases向量。最后在这个模型上面迭代计算就可以得到优化后的模型。

```python
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset) # embed.shape: (batch_size, embedding_size)
  # Compute the softmax loss, using a sample of the negative labels each time.
  loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, num_sampled, vocabulary_size))

  # Optimizer.
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities 
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
```

有了模型之后，我们该如何计算相似度呢？使用之前的相似度计算公式，我们可以先归一化embedding，使用valid_dataset的词向量与所有其他单词的词向量相乘，然后从大到小排序就可以得到按相似度排序的其他相似词。

```
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings)) # similarity.shape: (valid_size, vocabulary_size)
```

有了这个模型之后，我们就可以在模型上面进行迭代计算了。迭代的同时，我们每2000步输出一下平均损失值，每10000步的时候，输出一下和验证集相似的词。

```python
num_steps = 100001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1] # argsort will sort elements as ascended, so we need a minus symbol
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()
```

在模型计算好之后，我们可以使用TSNE降维方法，用二维表来表示我们的词向量，再绘制出来就可以得到最终的效果图片。

```python
num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :]) # ignore UNK

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
```

![Similar words in a graph](/attaches/dl-workshop-rnn-and-lstm/similar-words-in-graph.png)

### CBOW模型

现在到我们的动手时间了。我们将在上面的代码的基础上实现一个CBOW模型。

CBOW模型跟Skip Gram模型正好相反，在这个模型中，我们使用单词周边的单词去预测该单词。其模型如下：

![CBOW model](/attaches/dl-workshop-rnn-and-lstm/cbow-model.png)

这个算法的步骤如下：

- 随机生成一个大小为(vocabulary_size, embedding_size)的embedding矩阵（即所有单词的词向量矩阵，每一个行对应一个单词的向量）
- 对于某一个单词（中心词），从embedding矩阵中提取其周边单词的词向量
- 求周边单词的词向量的均值向量
- 在该均值向量上使用logistic regression进行训练，softmax作为激活函数
- 期望logistic regression得到的概率向量可以与真实的概率向量（即中心词的one-hot编码向量）相匹配

对比CBOW的计算步骤和SkipGram的计算步骤，我们可以来一步步的修改代码。

下面的内容，请结合[代码](https://github.com/gmlove/dl-workshop-rnn-lstm/blob/master/5_word2vec-problem.ipynb)来进行阅读。

我们要修改的第一个地方是`generate_batch`函数。

- labels的定义，其大小不再是(batch_size, 1)而应该为(batch_size // num_skips, 1)
- batch的赋值，batch现在应该为中心词周边的单词
- labels的赋值，labels现在应该为中心词
- print输出，label时，label的长度应该为batch_size // num_skips

```python
data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size // num_skips, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[target]
    labels[i, 0] = buffer[skip_window]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8//num_skips)])
```

修改之后，可以通过这样的测试来验证程序逻辑。当原始数据为`['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']`时，以num_skips=2和skip_window=1来调用generate_batch之后，得到的batch为`['as', 'anarchism', 'a', 'originated', 'as', 'term', 'of', 'a']`，对应的label为`['originated', 'as', 'a', 'term']`

第二个要修改的地方就是模型构建过程。

- train_labels的定义，train_labels的大小现在应该为(batch_size // num_skips, 1)
- 在进行loss计算之前，我们需要计算同一个label的train_data的均值向量。可以参考tensorflow的API `tf.segment_mean`来实现。

```python
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size//num_skips, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset) # embed.shape: (batch_size, embedding_size)
  segment_ids = tf.constant([i//num_skips for i in range(batch_size)], dtype=tf.int32)
  embed = tf.segment_mean(embed, segment_ids)
  # Compute the softmax loss, using a sample of the negative labels each time.
  loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, num_sampled, vocabulary_size))

  # Optimizer.
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities 
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings)) # similarity.shape: (valid_size, vocabulary_size)
```

在修改完成之后，我们就可以看到最后的结果。我们生成的词向量图与SkipGram模型生成的类似。

后续会继续RNN和LSTM的部分。敬请期待！

本文基于google在udacity上面关于深度学习的[课程](https://classroom.udacity.com/courses/ud730)而来。主要参考资料来自于斯坦福大学的自然语言处理课程[cs224d](http://cs224d.stanford.edu/syllabus.html)。