---
layout: post
title: "让机器自己玩游戏"
date: 2016-12-05 13:12:47 +0800
comments: true
categories: 
- tensorflow
- AI
- Machine Learning
- Reinforcement Learning
- DQN
- Deep Reinforcement Learning
- DeepQ Network
---

大家好，这次要跟大家分享一个很炫酷的东西。我们要实现一个机器学习算法，这个算法可以通过观察屏幕，产生一系列操作，进而控制游戏，取得高分。

## 我们的目标

Atari是1972年成立的一家美国公司，主要做的是街机、家用电脑、家用游戏机。很多早期的经典游戏都是出自Atari，比如什么乒乓球、网球、各种弹珠游戏等等。我们今天要让机器来玩的游戏就是出自atari的游戏，名为breakout。这个游戏是基于乒乓球的玩法的一个游戏，与乒乓球不同的是，这个游戏可以由单人控制。相信只要是80后，肯定都玩过这个游戏。

![Break out game](/attaches/let-machine-play-games/breakout-game.png)

<!-- more -->

## OpenAI Environment

为了实现我们的机器学习算法，我们是否要自己实现一个这样的游戏呢？当然不必。

在去年11月的时候，由特斯拉汽车的创始人也是spaceX的创始人Elon Musk带头发起了一个叫OpenAI的项目，这个项目主要的目的是降低机器学习的门槛，让人工智能更容易上手和实现。于是他们做了一个开发库，可以方便开发者开发强化学习算法和比较、分享算法的结果。而这个项目在今年（2016）4月份就发布了第一个beta版本。

使用这个项目提供的开发库，我们可以方便的运行breakout这个游戏，并采集到相关的游戏数据。

通过下面的代码我们就可以运行一个breakout的环境，并随机进行操作来玩游戏。这里的breakout游戏每一局有5条命，分数累加。当球碰到顶上的墙壁砖块之后，砖块被击碎而消失，分数加一。虽然这样玩出来的得分是很低的，但是我们已经有了一个基础。
```python
import gym
env = gym.make('Breakout-v0')

for i_episode in range(20):
  observation = env.reset()
  for t in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
      print('Episode finished after {} timesteps'.format(t + 1))
      break
```

![Breakout in OpenAI](/attaches/let-machine-play-games/breakout-in-openai.png)

从上面的代码中，我们可以看到gym给我们提供的东西有：

- observation：这个返回的值就是屏幕的观察数据，是一个大小为(210, 160, 3)的三维矩阵，里面是每一个像素的rgb颜色值
- reward：得分，当球撞击到墙壁砖块之后，返回1，否则返回0
- env.action_space：我们可以进行的所有操作，这里将会返回一个Discrete(6)，即离散的6个操作，大家可以认为这6个操作想象为，左加速移动，右加速移动，不动等
- env.step：执行一个操作，这个操作的值为0-5，即action_space中的某一个值

当我们有了这些数据的时候，我们事实上是有了什么呢？

- 当前游戏世界的状态
- 我们可以做的操作
- 一系列操作之后，我们可能会得到奖励，也就是得分

我们要解决的问题是什么呢？

问题就是，为了获取最大得分，我们如何在每一步选择正确的操作，即如何生成一系列的操作。关于操作的选择，就是策略。前面代码里面其实也是有策略的，我们的策略就是随机策略。我们的问题其实就是求最佳策略。

下面将向大家介绍强化学习的相关理论知识。

## 强化学习

### 马尔可夫决策过程

请看这个图：

![Markov Decision Process](/attaches/let-machine-play-games/mdp.png)

这张图里面的过程我们就将其称为马尔可夫决策过程。

在马尔可夫决策过程中，有一个agent，它可以观察environment，并操作environment，environment在接受操作之后，会按照一定概率产生一个状态改变，并可能会有给agent奖励。当然environment可能自己有自己的运作规则，自己也会不停的产生状态改变。

### 数学模型

我们可以定义马尔可夫决策过程中的一些概念如下：

- S: 一个所有可能的状态集合，在breakout这个游戏中，状态就是屏幕的像素值集合
- A: 一个所有可能的操作的集合，在breakout这个游戏中，就是action_space中指明的6种操作
- P: 状态转移的可能性，即关于状态空间的一个概率分布，给定一个状态，P可以表示该状态可能会转到哪一个状态，以及以什么样的概率转移到那个状态。结合breakout游戏进行理解就是，当我们捕捉到一个游戏画面的时候，下一个画面（即下一个状态）可能有多种可能，这多种可能性会形成一个概率分布
- R: 一个状态对应的奖励，有可能会跟随一个动作。需要说明的是，当有一个关键动作时，通常不会立即有奖励，大家可以想象一下breakout这个游戏，在球撞击顶部的墙砖时，就有正向奖励产生，而得分的前一时刻可能是不需要有动作的，或者说动作就是不做任何事

这里比较重要的是状态和动作的转换关系。从一个状态开始，可以选择多种动作，一旦选择了某一个动作，这个动作可能导致的多种下一刻的状态。所以动作-状态-动作会形成一种树形的关系，如下图所示：

![State action tree](/attaches/let-machine-play-games/state-action-tree.png)

这棵树的某一个分支就形成一局游戏。

有了这些定义之后，我们如何衡量某一个状态的价值，或者如何定义某一个动作的价值呢？

对某一局游戏，假设我们有一系列状态、动作和奖励，事实上我们可以定义状态和动作的价值如下：

![state action value formula](/attaches/let-machine-play-games/state-action-value-formula.png)

从这个公式中，我们可以看出，当前状态和动作的价值不仅跟当前的价值R0相关，还跟将来的价值相关。比如买股票，你今天的投资价值，不仅仅取决于今天股票的涨幅，之后每一天的增长都是今天的投资所决定的。所以我们可以知道，当前的状态和动作的价值是当前得到的奖励和将来的潜在奖励的和。

公式中还有一个参数γ，表示折扣因子。为什么要这个折扣因子呢？经济学里面有一句话"A dollar today is worth more than a dollar tomorrow!"，就是说今天的一美金比明天的一美金更值钱。我们可以这样理解，明天的价值是一个估计值，既然是估计值，那么就是不准确的，很有可能根本没这么多。对于钱而言，明天的钱还会贬值呢。所以在这个公式中，我们加上这个折扣因子，时间往后的价值需要乘以一个折扣因子的乘方，也就是说时间越久的价值对当前的价值贡献越小。

如果我们只对状态价值进行建模，我们可以得到这样的公式：

![state value formula](/attaches/let-machine-play-games/state-value-formula.png)

如何衡量一个状态的overall的价值呢？我们可以对每一种可能的游戏局求平均，也就是上面的公式的期望值。

![state value formula](/attaches/let-machine-play-games/expected-state-value-formula.png)

思考一下我们的问题，现在我们可以定义一个状态的价值了，一个状态的价值大，说明这个状态是一个好状态，我们就要让我们的策略向好的状态方面靠。似乎有了一些解决问题的思路了。

我们更进一步，从策略着手来分析这个问题。

**策略**就是一个从state到action的函数。也就是说，有了策略，我们就知道每一个状态下应该选择什么样的action了。我们通常用符号π来表示策略。

一个策略的价值函数就是：

![Value function for pi formula](/attaches/let-machine-play-games/value-function-for-pi-formula.png)

观察这个函数，我们可以发现这个函数其实可以写成迭代的形式。写成迭代的形式之后，我们就得到了著名的Bellman等式：

![Bellman Equation](/attaches/let-machine-play-games/bellman-equation.png)

有了这些之后，我们就可以定义最佳策略的价值为：

![Optimal Value function](/attaches/let-machine-play-games/optimal-value-function.png)

这个就是最佳策略的价值函数，它是所有可能的策略的一个最大值。写成迭代形式就是：

![Iterable Optimal Value function](/attaches/let-machine-play-games/iterable-optimal-value-function.png)

到这里我们似乎离问题更进一步了。但是，我们来考虑一下我们可以获取到什么。我们可以不断的用各种策略重复玩游戏，当我们的游戏局有相当的数量时，我们就可以近似得到状态转移函数，进而可以直接计算这个值。理论上这是可行的。但是实际上很难操作，特别是在状态太多的时候。而且就是可以得到最佳策略的状态价值，也不能直接得到最佳策略，需要经过计算才行。

我们再进一步考虑。能不能计算一个动作的价值呢？如果说可以计算在某一个状态下，某一个动作的价值，是不是就可以直接获取到最佳策略了？最佳策略就是选择动作价值最大的那个动作。事实上状态动作价值（或简称动作价值），我们一般称之为Q值，我们可以定义动作价值Q如下：

![Q value formula](/attaches/let-machine-play-games/q-value-formula.png)

写成迭代形式，也就是Q值的Bellman就是：

![Q value bellman formula](/attaches/let-machine-play-games/q-value-bellman-formula.png)

### 迭代算法

下面我们来看如何计算得到这个最佳Q值。

事实上我们只需要首先初始化每一个状态对应的价值为0，然后不停的玩游戏，进行迭代更新直到收敛就可以了。迭代算法如下：

![Q Value Iteration algorithm](/attaches/let-machine-play-games/q-value-iteration-algorithm.png)

对于一个状态空间比较小的MDP问题而言，我们直接用一个表来保存q值，然后迭代更新这个值就可以得到我们的最佳策略。这一算法是非常有效的。我这里不打算仔细分析这个算法。因为这个算法无法解决我们今天的问题。我们今天的问题状态空间太大了。所有可能的状态有`255^(260*160*3)`这么多种。

事实上，q值即是关于当前状态和动作的一个函数。既然这样，我们就可以想了，我们能不能用一个深度学习模型来模拟这个函数呢？当然可以，采用深度学习模型，我们可以将状态映射到一个动作价值上去。这就是深度强化学习啦！

## 深度强化学习

深度强化学习这个模型很早就有人研究，但是最早产生广泛影响力的是DeepMind公司在2013年发表的一篇论文。由于这篇论文的影响力，DeepMind也被google以数十亿美金收购。后来在2015年DeepMind进一步研究了这个模型，发表了一篇更清楚的论文：Human-level control through deep reinforcement learning。这篇论文就是讨论这个模型的。

这个模型使用的就是Q值迭代和神经网络的组合，所以算法全称为Deep Q Network，简称就是DQN。

核心算法如下：

![DQN Algorithm by DeepMind](/attaches/let-machine-play-games/dqn-algorithm-by-deepmind.png)

这个算法有这样一些关键步骤：

- 初始化深度学习模型，并在这个模型生成的策略下进行游戏
- 存储游戏产生的数据，包括状态、动作、奖励、下一刻的状态
- 从存储的数据中抽样数据进行迭代计算，更新模型

这个算法用到的神经网络模型如下：

![DQN Network Structure](/attaches/let-machine-play-games/dqn-network-structure.png)

这个卷积神经网络的参数如下：

![DQN Network Parameters](/attaches/let-machine-play-games/dqn-network-parameter.png)

关于卷积神经网络的资料，请大家参考斯坦福大学的cs231n课程。

我们只看一下这里的损失函数。回想一下我们之前的Q值迭代公式。从公式中我们可以看出，我们的最佳Q值就是最大的下一步的最佳Q值加上执行动作之后立即得到的奖励的均值。事实上当模型最终收敛时，平均起来看，当前时刻的Q值就等于下一步的Q值加上执行动作之后立即得到的奖励。于是我们只需要让当前时刻的Q值去逼近下一时刻的Q值和立即奖励之和。于是就有了我们的损失函数：

![DQN Network Loss Function](/attaches/let-machine-play-games/dqn-network-loss-function.png)

### 探索

在上面的伪代码中还可以看到，在玩游戏时，我们会以一定概率随机选择action进行执行。为什么需要这样呢？

大家可以想象一下，如果没有这个随机，会出现什么情况？在我们的算法迭代一段时间之后，算法发现了一个不错的策略，之后，可能就一直按照这个策略来生成动作了。这是有问题的，因为可能有很多潜在的表现更好的策略我们可能根本没有尝试过。加上这个随机过程，就使得我们有了探索策略空间的能力。

## 代码

接下来我们来分析代码。

这里我们使用tensorflow框架来实现这个算法，创建网络的代码如下：

```python
  def build_dqn_network(self):
    self.w = {}
    self.t_w = {}

    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
      32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
    self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
      64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
    self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
      64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

    shape = self.l3.get_shape().as_list()
    self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

    self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512,
      activation_fn=activation_fn, name='l4')
    self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')
```

网络一共四层，和前面的图里面的一一对应。

```python
  def build_optimizer(self):
    self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
    self.action = tf.placeholder('int64', [None], name='action')

    action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0,
      name='action_one_hot')
    q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1,
      name='q_acted')

    self.delta = self.target_q_t - q_acted
    self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')

    self.optim = tf.train.RMSPropOptimizer(
      self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
```

以上代码是关于loss函数的计算。可以看到，我们首先通过`self.q * action_one_hot`相乘得到当前action对应的q值，然后通过reduce_sum将其转化为一个数值。loss函数就是它和真实值的平方差。


```python
  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()
```

以上代码是自动玩游戏时的操作。可以看到，当游戏状态迁移之后，我们会保存当前的游戏数据。并在玩的局数到一定大小之后，就开始迭代更新网络。

```python
  def q_learning_mini_batch(self):
    s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
    q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
    target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, q_t, loss, summary_str = self.sess.run(
      [self.optim, self.q, self.loss, self.q_summary], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t
    })
```

以上代码显示了网络更新过程。我们先根据当前模型计算出下一时刻的q值，然后在discount之后计算和当前的奖励之和，作为目标q值，这个值将被传入optimizer计算图进行计算。

完整的详细的代码请参考这里：https://github.com/devsisters/DQN-tensorflow

需要说明的是，这里的模型收敛是很慢的，通常需要10-30个小时才能收敛到一个好的状态。

## 总结

今天我们看到了深度神经网络和强化学习结合起来的例子。这个例子的实现，可以给我们很大的启示。看起来DL+RL让机器拥有了某种智能。这就是AI。可以说深度学习和和强化学习就实现了AI。

当前有很多对这个模型的优化措施，包括加快泛化速度，以及让一个游戏的学习结果泛化到其他游戏中去等。是当前的一大热门模型。

关于强化学习还有很多理论知识未涉及，如果想要完全理解文章里面的公式及算法，请参考伯克利大学的强化学习课程。

### 参考资料

本文参考资料都来自于互联网，主要的资料如下：

1. 知乎专栏系列文章--DQN从入门到放弃：https://zhuanlan.zhihu.com/p/21421729
2. 知乎专栏系列文章--150行代码实现DQN算法玩CartPole：https://zhuanlan.zhihu.com/p/21477488?refer=intelligentunit
3. 伯克利大学强化学习课程：[CS188 from Youtube](https://www.youtube.com/channel/UCTmAYxRV7H9NTdgC9bNixvw)
4. devsisters的代码实现：https://github.com/devsisters/DQN-tensorflow




































