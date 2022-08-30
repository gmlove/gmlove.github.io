---
title: Rust内建最佳实践
categories:
- rust
tags:
- 性能
- 高性能
date: 2020-03-23 20:00:00
---

在前面的文章中提到我们在一个高性能场景中尝试了`rust`，那么它的效果如何呢？

在这次`rust`的尝试中，我们实现了一个通用的特征数据处理框架，并实现了几个常用的算子。这个数据处理框架目标是实现 `Spark ML pipeline` 的在线计算，为机器学习模型的在线推理场景提供特征处理。

我们选用了两个`rust`的`grpc`框架对外提供服务。它们分别是[`grpc`](https://github.com/stepancheg/grpc-rust)和[`tonic`](https://github.com/hyperium/tonic)，前者是基于线程池的实现，后者是基于rust异步模式`async/await`的实现。实验过程发现两者性能相差不大，`tonic`稍好，快`2ms`左右（不到5%），这可能是由于其数据结构设计更为精简带来的。

为了更有参考性，我们直接进行端到端的测试（用`grpc`客户端发起请求，在客户端采集数据），并与`scala`版本的实现进行性能对比。下面的结果中，服务端应用均部署在同一台`64核心`+`32GB内存`的服务器上，客户端也在此服务器上发起请求。由于数据处理的逻辑一致，客户端使用同一个`java`版本的实现。

<!-- more -->

对于`rust`版本的实现，为处理1000条数据，我们发起了20个并发请求，每个请求50条数据。客户端测试得到的响应时间在`25ms`到`38ms`间波动。观察服务端应用的内存占用，发现其稳定在`2%`，整个过程，几乎没有变化。`cpu`一直处于`1300%`到`2000%`之间。

考虑到实现的通用性，我们实现的`rust`版本的数据处理逻辑与`mleap`版本的数据处理逻辑是类似的。对于`mleap`，我们使用了官方推荐的[基于行计算](https://mleap-docs.combust.ml/mleap-runtime/row-transformer.html)的最高性能模式。`mleap`版本在处理同样的一组数据时，在没有发生`gc`的情况下，性能相对稳定，在`90ms`到`130ms`区间波动。如果某一时刻发生`gc`，则性能可能下降到`150ms`到`200ms`，甚至也有数百毫秒的时候。

再比较一下我们自实现的一个专用的java版本（将数个算子的计算过程进行联合优化，不仅针对性的优化了算法，还尽可能的减少了内存分配）。这个版本由于计算量降低了很多，其性能是很不错的，正常时，响应时间可以稳定在`20ms`到`40ms`间，但是我们也常常能观察到`70ms`及以上的波动发生。如果我们开启10个线程，每个线程处理100条数据，则数据相对更为稳定，但响应时间会增加`10ms`左右。

通过数据对比，`rust`不仅显示出了很高的性能，还表现出了特别强的稳定性。这都跟`rust`的极小的运行时设计及内存管理机制是分不开的。

`rust`除了这两点具有特别的吸引力之外，还有哪些地方值得推崇呢？事实上，`rust`在语言设计层面做了很多努力，以期帮助开发者直接采用当前社区所推荐的最佳实践，并在语言层面直接避免许多潜在的问题。我们的实践过程也从这些特性中受益颇多。下面将分享一些`rust`语言内建的一些最佳实践。

## `rust`语言内建最佳实践

### 摒弃类继承，仅支持`trait`多态

在`rust`中，我们没有`object`或`class`关键字，与之相对应的是`struct`，结构体。我们通过`struct`来组织数据，并在这些数据上面定义方法。一个`struct`即为一个类。但在`rust`中我们却无法定义一个结构体去继承另一个结构体的成员和方法。

作为面向对象编程思想的一个重要特性，继承，近年来越来越受到大家的质疑。继承是一个很好的解决代码复用问题的手段，但却常常被滥用，而我们总是很轻易的就滥用了继承。这些滥用表现在：1. 在设计时抽象出了没必要的复杂继承树；2. 为了代码复用，随意的（不合理的）把某些子类方法放到父类中；3. 重构时，简单的插入中间抽象类，导致越来越深的继承树；4. 单纯为了复用代码而抽象出并不合理的继承关系。

对于继承所带来的代码复用优势，当前我们所更为推荐的做法是：1. 通过接口实现来表达对象具有某一特性，并借此实现多态（这也是函数式编程所采用的做法）；2. 利用 **组合优于继承** 的思想设计职责更单一的组件和并实现代码复用。

在`rust`中，我们无法定义结构体的继承关系，我们却可以轻松的去定义一个`trait`，这里的`trait`可以类比`scala`语言中的`trait`，或者`java`中的`interface`，是一种更单纯的无状态的接口定义。同时，像在其他语言中一样，我们也可以在`trait`中提供默认的方法实现。

### 模块、属性默认`private`

在`rust`中，我们可以在一个文件中使用`mod`关键字定义一个或多个模块。模块内部可以存在各种语言支持的元素，如`struct` `enum` `constant` `trait` `function`等。这样的便利性，让我们可以更自由的以领域为中心去组织代码，从而提升代码的内聚性，降低耦合度。并且我们不用担心像`java`一样默认一个类对应一个文件，从而导致过多的文件。

同时`rust`在设计上默认限制访问方式为`private`，即仅供模块或结构体内部访问。这里的`private`限制包括：1. `struct`的内部属性默认无法从外部访问；2. `struct`的方法，默认无法从外部调用；3. 模块内部所有的元素默认均无法从外部访问；4. 模块默认可以访问其父模块的元素。

由于默认的`private`访问限制，`rust`程序将极大减少对外暴露非必须的接口，从语言设计层面促进了高内聚低耦合的特性。

### 数据默认不可变

默认情况下，`rust`中的数据都是不可变的，如果要使得数据可变，我们需要额外添加关键字`mut`，来显示的指定其可变性。

我们知道函数式编程特性中最重要的一点就要算不可变性了。正是由于数据不可变，我们可以轻松的在多线程中共享这些数据，可以轻松的实现惰性优化，或通过适当的重复计算来实现自动故障处理等。数据不可变还常常带来纯函数的特性，从而使得代码更易于理解。

`rust`在语言级别对数据的可变性提供了支持，除非我们显示的标记某一变量为`mut`可变，否则我们将无法修改其内部数据。`rust`提供的不可变标记比其他语言提供的不可变性要更严格，它真正表达了一组无法改变的状态。在`scala`或`typescript`中，我们可以通过`val`或者`const`来定义不可变的变量。但是它们仅仅标记为对应的变量不可重复赋值。我们依然可以改变变量对应的对象的内部状态。而在`rust`中，如果我们尝试这么做，我们的代码将连编译都无法通过。

### 实现错误处理的精致语法糖

在`rust`中，我们没有类似`java`一样的异常处理手段。比如，我们没法新建一个异常对象，然后`throw`到更上层。我们当然也没法`catch`住异常，而进行不同的处理。

`rust`提供了一种类似`c`语言的异常处理机制，即，通过函数调用结果来返回异常数据。这里可能有人会担心我们代码写得像`c`语言一样，遇到异常就要加一个`if`判断语句进行处理。这里的担心是多余的，`rust`语言在设计上专门为异常处理进行了特别的设计。

由于`rust`内建了强大的类型系统，所以，如果有异常发生，我们将会得到一个枚举类型的`Result<T, E>`值，它可能有`Ok(T)`或`Err(E)`两种情况。这时我们可以对返回的结果通过`match`语法进行类似`scala`提供的模式匹配进行处理。

但是，如果每个地方都需要`match`，那也将带来满屏幕的异常处理代码。`rust`是如何处理的呢？

其实我们平常处理的异常可以分为两类：1. 不可恢复异常；2. 可恢复异常。

对于不可恢复异常，通常是我们的代码写得不对，或者输入违反了某一个明确的假设，比如，越界访问一个数组就属于这种情况(`let a = [1, 2]; let b = a[2];`)。对于这个例子中的不可恢复异常，我们应当加入适当的越界判断逻辑，也就是说我们应该完善代码。这时通常的错误处理做法是，输出明确的被违反的假设，然后直接退出程序。`rust`为我们设计了`panic`宏方法以达到此目的。

对于可恢复异常，我们即可使用上述枚举类型`Result<T, E>`来进行处理。在我们的程序中，大部分异常都应当通过不可恢复异常进行处理。真正需要通过结果类型处理的异常会被限制到，比如文件读取错误，没有权限，自定义的必须要处理的异常等。

同时，对于我们常常需要调用的`match result { ... }`语句，`rust`提供了多种语法糖进行处理。如果我们需要`panic`，只需要调用`result.unwrap()`即可达到此目的，如果想要在`panic`时输出一些信息，则可以使用`result.expect("some message")`来实现。如果我们需要冒泡式的将异常抛出到上层进行处理，我们只需要在访问变量之前增加一个问号，即`let result = result?`，然后我们就可以在后续代码直接使用`result`变量了，就像没有异常一样。

总之，`rust`设计了非常精致而简洁的语法糖来支持异常处理，可以帮助我们编写健壮而简洁的代码。大家如想了解更多，请参考[这里](https://kaisery.github.io/trpl-zh-cn/ch09-00-error-handling.html)。

### 强大的类型推断和贴心的编译提示

用过`rust`的人，相信都会喜欢上`rust`强大的编译器，它的强大类型推断能力，可以让我们少写很多代码。

`rust`的编译器可以让我们尽量少做类型标注。

比如我们写下代码`let mut map = HashMap::new(); map.insert("abc", 123);`时，`rust`将自动的推断出`map`的类型为`HashMap<&str, i32>`。我们无需在定义`map`变量时指定类型。

再比如，当我们写下代码`let a: Vec<i32> = (0..100).collect();`时，`rust`自动为我们调用了生成`Vector`的函数。而当我们写下代码`let a: HashMap<i32, i32> = (0..100).zip((100..200)).collect();`时，`rust`又自动为我们调用了生成`HashMap`的函数。

喜欢`rust`编译器的另一个理由是其强大的发现错误的能力和贴心的编译提示。一个强大的编译器可以让很多错误提前暴露到编译期，以便我们可以更早的发现问题。谁也不想程序运行一段时间之后才报告有`bug`。尽管有时候`rust`编译器提示太多难免让人觉得沮丧，但我们最终总是会感谢它帮我们发现了很多低级的问题，节约了我们大量的时间。同时，我们也会感谢它推动了代码的风格一致性，编码的严谨性。

举个例子，由于`rust`具有不少内存访问规则限制，如果要人为分析变量的所有权（ownership），可能要让很多人望而生畏了。`rust`的编译器可以贴心的指出我们代码中的问题，比如，如果我们编写了如下代码，它涉及到引用的问题：

```rust
fn test() {
    #[derive(Debug)]
    struct B { b: i32 };
    
    let mut b = B { b: 10 };
    let b1 = &mut b;
    let b2 = &b;

    println!("b1: {:?}", b1); // 防止 b1 的生命周期提前结束
}
```

当我们尝试编译此代码时，我们将得到如下错误：

```
error[E0502]: cannot borrow `b` as immutable because it is also borrowed as mutable
  --> rust_test/src/lib.rs:70:26
   |
69 |                 let b1 = &mut b;
   |                          ------ mutable borrow occurs here
70 |                 let b2 = &b;
   |                          ^^ immutable borrow occurs here
72 |                 println!("b1: {:?}", b1);
   |                                      -- mutable borrow later used here
```

类似这样的地方还有很多，大家一上手便可以感受到。其实，`rust`的编译器不仅能清晰的指出问题，它还常常能给出我们要如何修改代码的建议。

比如，`rust`默认会在编译时检查变量是否使用过，对没有使用的变量会打印警告，并提示你`note: #[warn(unused_must_use)] on by default`，这时我们可能可以选择性的将这个编译选项关闭。

还比如，如果我们尝试格式化的打印一个没有实现`Debug` `trait`的`struct`（比如当上述代码中的`struct B`没有`#[derive(Debug)]`标记时），`rust`将拒绝编译代码，并提示`note: add #[derive(Debug)] or manually implement std::fmt::Debug`。

在遇到这类错误，并得到`rust`编译器贴心的`help`或`note`时，我们写代码也会感受到一丝丝温暖。

### 变量隐藏及强大的解构赋值

`rust`另一个让我觉得特别方便的地方是在同一个作用域内，我们可以定义重名的变量，这些重名的变量会隐藏掉之前的变量。比如，我们可以编写代码：

```rust
fn main() {
    let x = String::new("{\"x\": 1}");
    let x = parse_json(x);
    let x = x.x;
}
```

一般的静态类型语言都不支持在同一个作用域内定义重名变量，而是只支持父子作用域的同名变量隐藏。偏爱`python`的小伙伴会喜欢`rust`的这一特性，因为`python`的动态类型使得我们可以完成与上面类似的代码。

有人会担心同一个作用域内定义重名变量会带来不易读的代码，但是如果我们秉承小函数的思路，其实由于变量重名而带来的可读性问题基本不会发生。反而，我们常常要为属于不同的类型的同一个概念想出不同的名字，这让人很难受。

比如，上述代码在`scala`中，我们常常要给变量添加没必要的类型后缀，写成：

```scala
object App {
  def main(args: Array[String]): Unit = {
    val xJsonStr = "{\"x\": 1}";
    val xJson = parseJson(xJsonStr);
    val x = xJson.x;
  }
}
```

同时，提到赋值，不得不称赞的是`rust`强大的解构赋值功能。我们可以编写下面这样的代码：

```rust
fn main() {
    let p = Point { x: 0, y: 7 };
    let Point { x: a, y: b } = p;
    let (x, y, z) = (1, 2, 3);
    let ((a, b), Point {x, y}) = ((3, 10), Point { x: 3, y: -10 });
    // 在for循环语句中进行解构
    for (index, value) in v.iter().enumerate() {
        ...
    }
    // 在if-let语句中进行解构
    if let Some(color) = favorite_color {
        ...
    }
    // 在函数入参中进行解构
    fn print_coordinates(&(x, y): &(i32, i32)) {
        ...
    }
    // 在match语句进行解构
    match x {
        1 | 2 => println!("one or two"),
        3 => println!("three"),
        4..=10 => println!("four through ten"),
        _ => println!("something else"),
    }
}
```

### 其他

#### 使用引用传递，避免非预期的内存拷贝

除了一些拷贝成本极低的基本类型数据，`rust`内部总是使用引用传递数据，所以，我们无需担心非预期的拷贝。`rust`不会通过编译生成这样的拷贝内存的代码。如果我们要拷贝一个对象，我们需要显示的调用`clone`方法。

#### 推荐通过线程间通信来共享数据

解决线程间通信问题时，`rust`提供了用于线程间通信的`channel`模式，这与`go`的线程共享数据的哲学类似：`Do not communicate by sharing memory; instead, share memory by communicating.`

当然`rust`还支持通过传递变量所有权的方式，将变量安全的在线程间进行传递。

除此之外，`rust`还支持通过`Mutex`来实现共享对象的锁定访问。

#### 内建的测试支持

`rust`内建了对于测试的支持，不仅如此，`rust`甚至在设计层面有意识的区分了`单域测试`与`集成测试`两种不同的测试类型。这两种测试种类其实有着非常不同的属性。单元测试通常应该使用`mock`来构建，它应该非常快，测试到的路径足够多。而集成测试，通常为了验证模块之间是否能整合在一起工作，对于执行速度没有过于苛刻的要求。

对于单元测试，我们只需要在同一文件里面编写即可。而对于集成测试，我们需要放到一个单独的文件夹下面。

`rust`内建了常用的`assert`语句，同时还支持了在文档中编写的测试，即`doctest`（喜欢这一风格的`python`爱好者也将很乐意见到这样的支持）。

#### 统一的工程管理工具

`rust`提供了一套类似`npm`的统一的工程管理工具`cargo`。`npm`的使用，极大的促进了`javascript`生态的发展。同样，`cargo`让我们使用`rust`与使用`nodejs`一样简单，源代码组织形式一致，依赖管理便利。

其实，相比`npm`，`cargo`可以说走得更进了一步。`cargo`不仅提供了工程管理规范，甚至有关于文档的规范。只要我们按照`cargo`的规范去组织文档，那么我们运行`cargo doc`即可生成项目文档了。这一点跟`javadoc`类似。

#### 编译与方便的交叉编译

作为性能可以媲美`c`语言的高性能语言，`rust`将代码直接编译为机器码，并尽可能的将引用到的库进行静态链接。这一点跟`go`语言很类似。它极大的方便了我们对于程序的维护管理。在我们的实践中，一个编译好的二进制代码，可以在各种linux发行版中运行，无需安装其他依赖(`glibc`除外，除非我们编译`musl`版本)。类比`java`可知，我们至少需要安装`jre`，这带来了些许不便。如果我们愿意，我们甚至可以在一台裸的容器(通过`FROM scratch`创建)里面运行`rust`程序，除了系统内核，无需任何其他依赖支持。

`rust`不仅以二进制机器码为编译目标，而且支持广泛的运行平台。[这里](https://forge.rust-lang.org/release/platform-support.html)有一个列表。可以看到，我们甚至可以将`rust`程序运行在`android`或者`iOS`系统中。

同时，`rust`的交叉编译也是很方便的，我在`windows`上面，通过为数不多的几步操作，就可以用`llvm`编译一个`musl`版本的二进制可执行程序出来。

#### 高级特性支持

`rust`已经支持了很多我们所喜欢的高级特性，包括`async/await`的异步编程模式，元编程等。也包括很多较底层的特性，包括和`c`语言库的互操作性、内联汇编等。

## 编码时一些（烦人的）限制

`rust`为了达到的安全和高性能的设计目标，当然还是损失了一定的易用性的。相比`java` `scala` `javascript`或`python`这类高级语言，我们可能会在下面这几点中折腾挺长时间：

1. 编译期分配的内存大小必须编译期可知，使用基于`trait`的多态时，我们不得不利用`Box<dyn trait>`进行封装
2. 相比java，引入的概念更多，上手难度更高
3. IDE集成不够，调试体验较差，对`macro`编译期生成的代码支持较差
4. 即便有IDE支持也很难一次性通过编译

## 总结

总结起来，通过在线上机器学习推理的场景中尝试使用`rust`，我们发现`rust`在设计上拥抱了非常多先进的编程理念。在我看来，作为一个开发者，无论我们是否会在将来的项目中使用`rust`语言，这门语言都非常值得大家学习。它不仅仅是一门新的编程语言，更是一系列优秀的编程实践的集合，相信所有学习过`rust`的小伙伴都将有巨大的收获，也将潜移默化的指导我们以后编写的每一行代码。

`rust`无疑为高性能服务器编程提供了另一个选择，当前`rust`的发展可谓非常快速。但，`rust`可能还需要一个明星项目来为其背书，才能使其得到进一步的推广，让我们期待这样的明星项目的诞生。










































