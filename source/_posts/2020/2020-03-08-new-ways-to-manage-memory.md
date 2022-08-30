---
title: 内存管理的新思路
categories:
- rust
tags:
- 性能
- 高性能
- 内存管理
- gc
- 垃圾回收
date: 2020-03-08 20:00:00
---

在最近的一个客户项目上，为了做性能优化，我们花了大量的时间，然而最终结果还是不够理想。我们的场景是实现特征处理过程和机器学习模型线上推理服务。由于用户量巨大，我们需要做到2万的TPS，每个请求需要在30ms内返回，且每个请求中包括对1000个项目的处理过程。

我们所使用的技术栈是`spring`和`grpc`。在经过极致的代码优化及内存调优之后，运行在一台`32GB`内存`64核`CPU的服务器上，我们发现`90%`的请求可以在`25ms`完成。但是如果观察`99%`的分位线时，响应时间就下降到了`70ms`，有时候还可能超过`100ms`。

为什么会出现上面这么明显的波动呢？问题出在`java`的`gc`上。其实对于`gc`，我们已经非常仔细的做过调优了，整个过程没有`full gc`的发生。然而，在持续的压力测试下，`java`的`young gc`却在频繁的工作。由于处理的数据量过大，新生代的`gc`几乎每秒都会触发一次，每次释放`5GB`内存，耗时`30ms`左后。

<!-- more -->

由于要服务于线上上亿的用户群，这样的性能还是不够理想，难以直接交付给客户使用。

`java`语言发展了这么长时间，其性能还是能为人们所认可的。在一些性能测试上面，`java`几乎可以媲美`c++`的计算性能，比如[这里](https://benchmarksgame-team.pages.debian.net/benchmarksgame/fastest/java.html)就有一个这样的测试。然而内存管理却一直是`java`语言一个挑战，即便`java`已经有相对非常成熟的`gc`算法了。在编写极致性能需求的服务端程序时，`java`由于其本身的性能波动，似乎还是难以胜任。

究竟要如何做高性能的内存管理呢？难道非得像`c/c++`一样的手动去管理内存吗？

## 当前流行内存管理机制

我们先看看当前流行的编程语言所采用的内存管理机制。以我所接触过的，使用相对广泛的编程语言为例，可以整理如下：

- `c`: 手动进行内存的申请和释放
- `c++`: 通过`delete`指针，在析构函数中手动释放内存
- `java`及各种基于`java`的语言: 适时分代`gc`，多种垃圾回收算法
- `python`: 引用计数，自动触发适时分代回收
- `obj c / swift`: 引用计数或自动引用计数回收，编译期插入引用计数代码
- `go`: 适时分代gc，并行标记清除模式
- `javascript(v8)`: 适时分代gc，并行标记清除模式
- `php`: 以引用计数为基础，适时触发`gc`
- `lua`: 标记清除模式

可以看到，当前的自动内存管理机制以 **引用计数** 和 **分层并行标记清除** 为主。

如果内存释放及时，引用计数机制对于程序运行时性能影响会比较小。但每一个对象都需要分配额外的内存去跟踪引用数量，这带来了额外的内存占用。如果没有自动引用计数的机制，在编写代码时，手动管理计数会带来不小的额外负担，内存的及时释放取决于引用计数代码的正确性。

而对于标记清除式的内存管理，由于其不可避免的会带来程序暂停，且并行标记还需要占用cpu时间，会对程序性能产生较大影响。

我们的眼光投向了`rust`。`rust`是一门没有`gc`也不使用引用计数（不以此为主）进行内存管理的语言。那它究竟是怎样管理内存呢？它还能让我们像写`java`代码一样流畅吗？

## 内存管理的新思路

回顾一下日常编写代码的过程，在实现某一个函数时，有这样几个要素：入参 函数体 返回值。比如，假设我们有下面这个计算字符串长度的函数：

```rust
fn func() {
    let s1 = String::from("hello");

    let (s2, len) = calculate_length(s1);

    println!("The length of '{}' is {}.", s2, len);
}

fn calculate_length(s: String) -> (String, usize) {
    let length = s.len();
    (s, length)
}
```

考虑内存释放这件事，我们会希望：

1. 对于函数`func`中的变量`s1`，由于我们将其作为参数转交给了函数`calculate_length`，在后续的执行过程，我们就不再希望关心它对应的字符串内存了。
2. 对于`calculate_length`函数中的入参`s`，由于其作为函数值返回，我们不希望在离开函数时释放其对应的内存。
3. 对于`calculate_length`函数计算得到的`length`变量，我们同样不希望在离开函数时释放内存，因为它也是一个返回值。
4. 对于`func`从`calculate_length`函数得到的变量`s2`和`len`，我们希望在`func`函数离开的时候立即释放。

上述虽然只是一些很简单的想法，但这里面似乎隐藏着一些规律，我们能不能更进一步呢？甚至，我们是不是可以以此规律去设计一种新的内存管理方式呢？事实上，稍加抽象可以得到：

- 变量可以属于某个函数，即其所有权为某个函数
- 在函数结束的时候，回收函数所持有的所有变量的内存
- 在发生子函数调用时，如果我们传入某一变量，该变量的所有权将移交到子函数中去
- 在子函数调用返回时，如果有返回一些变量，则这些变量的所有权将回交给当前函数

这就是`rust`的内存管理基础。初次接触这种全新的内存管理方式时，不禁让人觉得眼前一亮。这种方式看上去自然而高效，且根本无需独立的垃圾回收器。

在`rust`中，我们实际上讨论的是更细致的值（内存）的所有权问题，但基本的观点与上述几点相似。更为严格的，`rust`中定义了如下几条关于值（内存）的所有权的规则：

- 每一个值均存在一个对应的变量，该变量是这个值的所有者
- 同一时间每个值只能有一个所有者变量存在
- 当所有者变量离开当前可访问的代码范围（可以是一个函数，或一个由括号`{}`定义的一个范围等）时，该值对应的内存将会被释放
- 变量作为函数参数值传递时，值的所有权将移交到函数参数对应的变量中
- 函数返回一个变量时，该变量对应的值的所有权将回到上层函数对应的变量中

## 延伸

实际上，我们在编写代码时，会碰到比上述场景复杂得多的场景。那么这几条原则是否还奏效呢？

其中一个我们很快会碰到的问题就是，由于存在所有权移交，我们可能需要每次函数调用都返回一些额外的值。如果每个函数都这么写，那可能是一场灾难。因为这将带来不够清晰的函数定义，而这种不清晰将侵入到整个代码库里面去。

`rust`是如何解决这个问题的呢？这就是`rust`中的引用和借用的机制。比如我们可以编写如下的代码：

```rust
fn func() {
    let s1 = String::from("hello");

    let len = calculate_length(&s1);

    println!("The length of '{}' is {}.", s1, len);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

通过在变量前面增加一个`&`符号，我们得到了一个称为 **变量引用** 的东西。对于变量引用而言，其对应的值的所有权并不会并发生移交。在函数调用时，我们可以仅仅将某一值的引用传递过去，这个时候发生的事情，称为借用。这里可以类比我们日常生活中借用别人的东西的场景，借用完之后，再归还给他。

另一个不难想到的问题是：如何去按照面向对象的方式组织数据？事实上，关于值的所有权定义也可以用来解决这个问题。在前面的讨论中，我们以函数作用范围举例，但这里的范围其实可以不只是函数，也可以是某个对象（或结构体，`rust`中成为`struct`）。也就是说，我们可以将某一个值作为另一个对象的一部分绑定到该对象上去，这样该值的所有者就变成了这个对象。这个对象本身可以作为一个值绑定到某一个变量中去，从而构成了一个完整的闭环。

还有一个问题，可能会成为`rust`的难题，即需要共享某一个值的场景。这时，我们需要让一个值同时属于多个所有者。这跟我们前面提到的`rust`的单一所有者原则相悖。在这样的场景下，`rust`定义了一个称为`Rc`的结构体，它存储的是这个对象的引用及一个引用计数。`Rc`实际上就是`reference count`的缩写。这里的内存管理其实就退化为引用计数式的内存管理。

我们还可以提出更多的问题，比如：多线程的情况下，值（内存）的所有权要怎么变化？如何实现线程同步的锁机制？如何处理循环引用问题？事实上，诸如此类的问题已经被`rust`的社区及其编写者们思考并实践了多年。`rust`所特有的所有权特性为解决这些问题提供了全新的思路，大部分问题也都被优雅的解决了。

## 总结

诞生于`Mozilla`社区的`rust`程序设计语言为我们带来了全新的内存管理思路。近几年来，`rust`语言发展迅速，由于其在性能和稳定性可以超越`c/c++`，在易用性上不输`java`等高级语言，在高性能服务器端开发领域`rust`已开始崭露头角。

`rust`是否可以解决我们的线上特征处理和模型推理的极致性能需求呢？我们正在尝试过程中，同时满怀期待。


参考：
- https://blog.codingnow.com/2018/10/lua_gc.html
- https://blog.devtang.com/2016/07/30/ios-memory-management/
- https://www.cnblogs.com/geaozhang/p/7111961.html
- https://segmentfault.com/a/1190000018161588
- https://juejin.im/post/5b398981e51d455e2c33136b
- https://doc.rust-lang.org/stable/book/

