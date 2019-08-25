---
title: 从改善设计的角度理解TDD (2)
categories:
- tdd
- 敏捷
tags:
- tdd
- agile
- 敏捷
- 测试
- 质量
date: 2019-08-18 20:00:00
---


在文章开始之前，我们先回顾一下TDD带来的好处。当我们理解TDD之后我们至少会发现下面这三点：

1. TDD是一种更加自然的编程方式，因为我们总是要先弄清需求再编写代码，而这跟TDD先写测试（通过测试清晰的定义需求）再写实现的顺序是完全一致的。
2. 先写测试还要求我们站在使用者的角度来编写测试，这样我们可以自然的驱动出来更好的设计。
3. 由于TDD天然的特性，测试在编写代码之前就有了，自然我们也就无需担心测试覆盖率不够带来的质量问题了。

TDD给我们编写代码带来的好处多多，我前面有一篇[文章](http://brightliao.me/2019/07/20/tdd-for-improving-design/)主要分析了如何从改善设计的角度理解TDD，相信大家能感受到TDD给改善程序设计带来的好处。这里我想再次分享一个用TDD改善设计的例子，我们将从中看到不用TDD时我们的代码可能会变成什么样，而用了TDD又将变成什么样。

<!-- more -->

## 背景

有这样一个需求，我们需要定义一个有向无环图将一组小的`Task`组织成一个功能更强大的`Job`，然后执行`Job`按照任务（`Task`）先后关系运行这个图。其实我们能找到很多用这样的方式来编排任务的开源工具，比如下面这些。

![Workflow工具](/attaches/2019/2019-08-18-tdd-for-improving-design-2/workflow-tools.png)

我们还希望以后能编写一个UI界面来辅助用户快速的编排任务，用户可以以拖拉拽的方式可视化的编排，可以灵活的配置任务参数，同时系统将自动的校验用户指定的及任务间传递的参数。

基本的需求就是这样，在了解了这些之后，我们来分别看看直接进行设计和用TDD进行辅助设计会是什么样子的。

## 直接设计

分析上述需求，如果我们想在用户编排任务的时候进行参数校验，那么我们必须在每个任务运行之前就有办法获取到各种参数和返回值的类型信息，它们将包括任务的可配置参数、任务的入参、任务的返回值等。由于这些参数都是针对某个具体的任务来定义的，我们可以设计任务的几个接口来获取这些类型信息。我们暂定只需要上述三种类型信息，可以得到如下的接口设计：

```java
interface Task {
    TypeOfSomeConfigurationType getConfigurationMeta();
    TypeOfSomeInputType getInputMeta();
    TypeOfSomeResultType getResultMeta();
    SomeResultType execute(SomeInputType input, SomeConfigurationType config);
}
```

上面的`SomeXXXType`应该是什么呢？为了支持灵活的配置，这里我们需要一个非常灵活的数据结构。稍加思索大家应该就能想到我们可以用`Map`。我们可以定义一个`Map<String, Object>`的类型来存储任意键值，这时候，对`SomeConfigurationType`而言，`key`是配置的名字，配置的值可以是任意的一个对象；对于`TypeOfSomeConfigurationType`而言，它应该是一个`Map<String, Class<?>>`类型，`key`与配置的名字一致，值用于描述配置的值的类型。其他的`SomeInputType`和`SomeResultType`均可以这样定义。到这里我们可能还会有点小小的自豪，问题很简单啊，引入一个`Map`数据结构就完美解决了。这个时候我们的`Task`接口会设计成下面这样：

```java
interface Task {
    Map<String, Class<?>> getConfigurationMeta();
    Map<String, Class<?>> getInputMeta();
    Map<String, Class<?>> getResultMeta();
    Map<String, Object> execute(Map<String, Object> input, Map<String, Object> config);
}
```

仔细观察上面的接口，好像没那么干净，但似乎也挑不出什么毛病。`Map`看起来对于`Task`的实现会有一些干扰，因为`Task`实现里面势必会引入一些必要的强制类型转换的工作。能否避免这样的比较脏的类型转换代码呢？我们思考了一下，可能可以提供一些Utilities工具类来进行一些支持，或者编写一个`AbstractTask`抽象类，将一些公共的逻辑放到抽象类中进行复用。总体上感觉问题应该出在需要支持 **灵活的配置** 这样的需求上。由于需求是要保证灵活性，我们就只能将接口定义成这样。

如何将任务组织成为一个有向无环图呢？这看起来也不是难事，先定义一个DAG，然后一个节点一个节点往里面塞数据就可以了。同时为了支持有向无环图的执行我们可以定义一个`execute`方法。这个时候我们的`Job`类大概会设计成下面这样：

```java
interface Job {
    void addTask(Task task, Task... subTasks);
    void execute(Map<String, Object> input, Map<String, Object> config);
}
```

到这里基本的设计就差不多完成了。我们会隐约感受到一些不够`clean`的设计，但是似乎难以控制。为了尽快实现功能，先这样定吧。

## TDD驱动设计

如果我们用TDD的思想来驱动开发，情况会是什么样呢？

按照TDD解决问题的思路，我们要先编写一个测试。那么第一个测试要怎么写呢？分析一下需求可以发现，我们的目标是要实现一个DAG任务运行框架。那么我们的测试目标应当就是这个框架。这个框架需要实现的功能是：

1. 允许用户自定义一些`Task`；
2. 允许用户将这些`Task`组织成一个有向无环图；
3. 提供接口运行这个图。在测试的代码中，我们需要模拟这个框架的使用方式，提供输入，然后验证输出。

于是这个测试的实现就会包含这样几个步骤：

1. 定义`Task`；
2. 将`Task`组织成一个有向无环图；
3. 利用框架运行这个图；
4. 检查运行结果是否是我们想要的。

我们可以先编写测试模板代码如下：

```java
public class JobSchedulerTest {
    @Test
    public void should_run_the_defined_job_and_output_the_expected_result() {
        // define tasks
        // create dag
        // execute dag
        // verify execution result
    }
}
```

如何定义`Task`呢？`Task`需要保持足够的灵活性，以便可以支持任意可能的配置，同时`Task`要暴露接口让框架获取到输入、配置以及输出的元数据信息（类型信息）以便支持参数校验。由于我们使用`Java`这样的静态类型语言，事实上我们可以认为这些元数据信息是自动提供的。那么一个理想的`Task`就应该定义成下面这样：

```java
public class SimpleCalculationTask {
    public CalculationResult execute(CalculationInput input, CalculationConfig config);
}
```

这样的`Task`定义我们应该会比较满意，因为它只有一个接口，并且这个接口非常简洁而又有明确的类型信息。要实现这样的接口也应该是非常容易的。比如在测试中我们可以设计一个非常简单的加法任务，它不仅要输出加法的结果，还会附带输出获取到的输入参数，同时我们可以配置加法在什么区间可以按正常工作，而在另一个区间直接输出`-1`。这样的加法任务可以实现如下：

```java
public class AdderTask {

    public static class AdderInput {
        public int left;
        public int right;
        public AdderInput(int left, int right) { this.left = left; this.right = right; }
    }

    public static class AdderConfig {
        public int maxSupportedValue;
        public int minSupportedValue;
        public AdderInput(int maxSupportedValue, int minSupportedValue) {
            this.maxSupportedValue = maxSupportedValue; this.minSupportedValue = minSupportedValue;
        }
    }

    public static class AdderResult {
        public int left;
        public int right;
        public int result;
        public AdderInput(int left, int right) { this.left = left; this.right = right; this.result = result; }
    }

    public static class AdderInput {
        public int left;
        public int right;
        public AdderInput(int left, int right) { this.left = left; this.right = right; }
    }

    public AdderResult execute(AdderInput input, AdderConfig config) {
        int result = input.left + input.right;
        return result > config.maxSupportedValue || result < config.minSupportedValue
                ? new AdderResult(input.left, input.right, -1)
                : new AdderResult(input.left, input.right, result);
    }
}
```

有了这样的定义之后，我们需要想办法将任务组织成一个有向无环图。怎样才能方便的构造这个图呢？最好有一个调度器，可以在某个任务后安排其他任务，也可以安排几个并行执行的任务，而我们需要提供一种方式来构造下游任务的参数。按照这样的想法可以得到下面这样的测试代码：

```java
public class JobSchedulerTest {
    Job createJob() {
        TaskScheduler scheduler = new TaskScheduler();
        Job job = scheduler.scheduleTask(new AdderTask())
            .next(new AdderTask(), adderResult -> 
                new AdderInput(adderResult.left, adderResult.result));
        job = scheduler.scheduleParallelTask(job, new AdderTask())
            .next(new AdderTask(), (jobAdderResult, taskAdderResult) -> 
                new AdderInput(jobAdderResult.result, taskAdderResult.result));
        return job;
    }

    @Test
    public void should_run_the_defined_job_and_output_the_expected_result() {
        Job job = createJob();
        // execute dag
        // verify execution result
    }
}

```

在写完这些代码的时候，我们发现设计更为丰满了。其实到这里主要的设计就可以算完成了，后续的代码在逻辑上比较直接，为避免赘述我们就不继续了。

## 回顾

回顾两种方式产生的设计，不言而喻，TDD驱动出来的设计比直接设计出来的设计要易于使用得多，而且干净得多。不知道大家有没有被TDD给我们带来的强大能力所惊艳到呢？至少我是被惊艳到了。除了设计在易用性和干净程度上，事实上上面直接设计出来的东西还忽略了一部分功能，试想在运行时，我们如何将任务的参数进行上下串联呢？这是因为如果我们直接开始设计，由于一开始很难考虑清楚所有的情况，我们总是容易漏掉一些东西。而TDD由于是先从使用上面来定义设计应该有的接口和互操作方式，这使得我们更不容易漏掉这样的关键步骤。

我们经常听到说TDD可以带来十倍效率提升，一开始有人一定会觉得这有点夸大，难以相信。但是在这个例子里TDD带来的收益可以说是非常巨大，说十倍效率提升我觉得也不过分。我们可以尝试比较一下。如果我们用直接设计的方式得到最终的设计，那么后续我们在开发和调试`Task`的时候，我们将面临这样的问题：

1. `Task`的实现代码中充斥着大量的类型检查和强制类型转换；
2. 阅读代码时，如果没有文档，我们无法知道某个`Task`究竟需要怎样的参数，因为全都是`String`和`Object`，必须要将`Task`运行起来通过断点的方式才能知道到底是什么；
3. 如果有某个`Task`的详尽的文档，我们还需要维护文档和实现的一致性。

单论这三点，我们就将消耗大量的时间。而用TDD驱动出来的对使用者友好的设计，类型检查在创建DAG的时候就可以自动完成，我们也无需维护文档，代码也更清晰可读。而TDD带来的测试质量保护网给我们节省的时间就更不用说了。

我们再次来回顾一下TDD的实践过程。由于TDD要先写测试，这就使得我们不得不在写代码之前（设计之前）先从使用者的角度思考如何使用（如何设计），因而我们也就自然的得到了易于使用的（好的）设计。上面的思考过程可以印证TDD如何驱动出更好的设计的过程，我们先后从使用者的角度思考了如何定义`Task`，从使用者的角度思考了如何构造DAG。这是两个非常关键的设计，这两个设计将给后续实现新的`Task`，调试`Task`带来巨大的便利。

当我们熟练使用TDD之后，实践TDD其实是一件非常愉快的事情。你将清晰的定义出需求，自然的写出更好的代码，同时TDD天然的测试屏障给我们的每行代码的正确性都带来了信心。我们不再会直接写出100行没有测试的代码，然后心里非常虚，然后还需要通过端到端启动应用这样非常低效的方式来验证代码正确性，而在修改了代码之后我们还不得不重复这样枯燥无聊的步骤。TDD可以让我们变成一个快乐的程序员。

最后，我想重复一下第一篇[文章](http://brightliao.me/2019/07/20/tdd-for-improving-design/)最后一段的内容：

通过上面的经验的分享，不知道大家是不是更认可和接受TDD了呢？但是要熟练运用起来，关键还是在于刻意的去练习。每天的日常工作都是机会，希望大家能保持开放的心态，严格要求自己，遇到问题多讨论交流。当团队中所有人都会TDD，代码能力都上去了的时候，我们才能说我们是一个高效的团队，我们能做高质量的产品。所以，加油吧！











