---
title: python性能优化二三事
categories:
- python
- performance
tags:
- GIL
- python
date: 2019-07-25 20:00:00
---

随着机器学习的流行，Python近几年的热度一直在上升，再加上Python本身语言设计的简洁直观和易用，Python越来越得到开发者的青睐。但是我们却时常听说Python性能低，不如java，更比不上C。在这些抱怨背后到底是什么原因呢？Python真的性能低下吗？有没有什么优化的办法呢？

对于单纯的复杂计算过程，Python性能是比较低的，这是由于Python本身在设计时首要考虑的是如何快速完成工作（get things done），所以在性能上难免会有一定的牺牲。但是由于python和c有着非常好的互操作性，这类问题都可以通过实现一个c语言的版本来解决。当然从代码编写技巧的角度也有一定的优化空间，如果我们想做极致的性能优化，可以参考[官方的性能优化技巧](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)。

<!-- more -->

多数时候，当我们想加快程序运行速度，使用多线程或多进程并行应该是首要考虑的方案。它将能有效利用资源，直接带来数倍至数十倍的性能提升。然而Python的多线程可以说饱受诟病。有经验的Python开发者可能会说Python的多线程就是鸡肋，多进程才能真正带来计算加速。这是为什么呢？本文将进行简单的分析，并分享我们实际项目中的性能优化经验。

## Python多线程问题

(下面所有测试使用Windows系统下的Python 3.7.2版本进行，计算机有4核心cpu)

假设我们有这样的一段这样的密集计算型代码：

```python
@log_time
def heavy_calculation():
    import math
    a = 0
    pow = math.pow
    for i in range(10000000):
        a += pow(2, 10)

@log_time
def exec_in_single_thread():
    heavy_calculation()
    heavy_calculation()

@log_time
def exec_in_multi_thread():
    from threading import Thread

    threads = [Thread(target=heavy_calculation), Thread(target=heavy_calculation)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

exec_in_single_thread()
exec_in_multi_thread()
```

（完整代码可以参考[这里](https://github.com/gmlove/experiments/tree/master/python_perf)）

运行上面的代码可以发现`heavy_calculation`执行一次需要2s左右，`exec_in_single_thread`花费4s，`exec_in_multi_thread`使用了多线程，但是居然也要花费4s！再仔细观察结果，就会发现，同是`heavy_calculation`，在`exec_in_single_thread`中花费2s，但是在`exec_in_multi_thread`中居然要花费4s，而且两次调用均花费了4s。

## Python的GIL问题

多线程看起来确实并不能有效利用多核进行加速。这是为什么呢？答案是python的GIL问题。Python这门语言其实有很多解释器实现，除了最流行的c语言实现CPython，还有java实现Jython，甚至Python自身实现的PyPy。GIL问题目前在CPython和PyPy中存在，Jython没有这个问题。

GIL的全称是Global Interpreter Lock，即全局解释器锁，从官方的[介绍](https://wiki.python.org/moin/GlobalInterpreterLock)中我们可以了解到引入它是由于CPython的内存管理是非线程安全的，需要避免多个线程同时去执行代码。到这里大家就明白了，python的多线程无法用来做计算加速！

不过，看起来如果重新用一种线程安全的方式来实现CPython的内存管理就能解决问题了，不是吗？但是现实问题远非这么简单，因为Python现在有大量的库的实现都依赖这个GIL，也就是没有考虑线程安全问题。这同时也导致了python的库的兼容性问题，比如虽然Jython没有GIL，但是在运行时它可能会有未知的线程问题，所以就难以流行起来。更多相关的问题可以参考GIL的官方介绍文档来了解。我们这里主要分享一下当我们遇到这个问题的时候要如何解决。

## 优化Python程序性能

其实我们不能直接说python的多线程是无用的，这还得看我们的具体问题。如果我们是想同时执行一个cpu密集型和一个io密集型任务，那么python的多线程依然是有效的。比如执行下面的测试我们将看到多线程带来了速度的提升：

```python
@log_time
def heavy_calculation():
    import math
    a = 0
    pow = math.pow
    for i in range(10000000):
        a += pow(2, 10)

@log_time
def heavy_io():
    open(r'some-600MB-file', 'rb').read()

@log_time
def exec_in_single_thread():
    heavy_calculation()
    heavy_io()

@log_time
def exec_in_multi_thread():
    from threading import Thread

    threads = [Thread(target=heavy_calculation), Thread(target=heavy_io)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

exec_in_single_thread()
exec_in_multi_thread()
```

（完整代码可以参考[这里](https://github.com/gmlove/experiments/tree/master/python_perf)）

这里的IO操作差不多需要0.5s，执行后会发现`exec_in_single_thread`花费了2.5s左右，而`exec_in_multi_thread`只花费了2s。

当我们想要并行加速多个计算密集型任务时，主要思路有两个：

1. 用c语言实现，显示的释放GIL，之后就可以利用线程加速了
2. 改用多进程来加速，避免了GIL问题

下面将分别介绍这两种方案。

### 显示的释放GIL加速

参考Python的[文档](https://docs.python.org/3.7/c-api/init.html#thread-state-and-the-global-interpreter-lock)我们知道，其实可以很简单的在c语言中用一个宏来实现GIL的显示控制。对于上面的计算，我们可以用c语言实现如下：

```c
static PyObject *
demo_pure_heavy_calculation(PyObject *self, PyObject *args)
{
    long a = 0;
    for (int i = 0; i < 10000000; i++) {
        a += pow(2, 10);
    }
    return PyLong_FromLong(a);
}
```

（完整代码可以参考[这里](https://github.com/gmlove/experiments/tree/master/python_perf/c_extension)）

我们编译运行此代码，将会看到执行时间为0.028s左右，c语言的实现带来了近百倍的提速。但是由于我们还没加入GIL的释放代码，所以多线程运行时，速度并不会加快。

修改代码如下：
```c
static PyObject *
demo_heavy_calculation_allow_thread(PyObject *self, PyObject *args)
{
    long a = 0;
    Py_BEGIN_ALLOW_THREADS
    for (int i = 0; i < 10000000; i++) {
        a += pow(2, 10);
    }
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(a);
}
```

改成这样，再次运行，我们将看到多线程带来的提速了。

### 更简单的实现

上面这样的实现能解决问题，但是看起来略繁琐，其实我们有一个简单的库`Cython`可以辅助我们更简单的编写代码。参考[官方文档](http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html)，我们可以用类python的语法编写代码如下：

```python
from libc.math cimport pow

def heavy_calculation():
    cdef double a = 0
    cdef int i  # 这里的定义使得cython编译器会优化下面带range循环为c的for循环
    with nogil:
        for i in range(10000000):
            a += pow(2, 10)
    return a
```

（完整代码可以参考[这里](https://github.com/gmlove/experiments/tree/master/python_perf/cython_extension)）

Cython运行时会把上述函数编译为跟我们上面差不多的c语言代码，大大简化了我们的代码维护工作。同时简单的使用`with nogil`就实现了GIL锁的释放。并且测试还会发现，用Cython的实现，几乎会比用c实现的快2-3倍。这应该是编译优化导致的。

如果我们的代码是纯c实现，不需要操作python对象，那么我们还可以参考[SWIG](http://www.swig.org/tutorial.html)，它给我们的代码管理也带来了方便。

### 使用多进程加速

为了避免重写代码，最简单的恐怕是直接改为多进程的机制。

如果当前是用线程池实现的，改为使用python的进程池，我们几乎只用修改一行代码就可以实现。两种实现方式示例如下：

```python
def execute_concurrently(num_workers, func, *parameters_list):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        func_future = {executor.submit(func, *parameters): parameters for parameters in zip(*parameters_list)}
    for future in concurrent.futures.as_completed(func_future):
        try:
            data = future.result()
            yield data
        except Exception as exc:
            logger.warn('task(%s) generated an exception: %s' % (func_future[future], exc))

def keep_silent_on_exception(func: Callable, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warn('{} raises exception: {}'.format(func, e.args))
        return 'keep_silent_on_exception'

def execute_concurrently_by_process(num_workers, func, *parameters_list):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for data in executor.map(keep_silent_on_exception, [func] * len(parameters_list[0]), *parameters_list):
            if data != 'keep_silent_on_exception':
                yield data

```

上述代码进行非常简单的封装，并忽略了错误。

只是这样就够了吗？改为进程池，带来的问题就是所有的参数传递，都变为了进程间的数据传递，也就是进程间通信问题。

参考Python的进程池文档可以知道，Python内部实现是先使用`pickle`将数据序列化，然后再相互传输的。这就提醒我们：

1. 在设计并发函数参数的时候，需要特别注意参数，尽量避免将大量的数据作为参数进行传递，如一个大型的`numpy`数组。
2. 我们无法将不能`pickle`序列化的对象(如某一个对象，其内部有一个类型为socket连接的属性)作为参数传递。
3. 我们需要深入分析多进程情况下的对象生命周期，比如，多进程可能导致我们为每一个进程创建了一个socket连接，这会不会带来问题呢？

在实践过程中，还有以下两点可能是值得考虑的：

1. 避免在将要循环执行的函数内部执行某一可共享的耗时操作，比如在上面代码中传入的`func`的实现里面就不宜加入耗时且可共享的操作，这个时候我们可以使用延迟初始化的方式来解决这个问题。比如我们可以设计一个单例的`SharedObjects`对象，然后在其中延迟进行这种操作。

```python
class SharedObjects:

    def __init__(self):
        self._some_big_object_from_heavy_io = None

    def get_some_big_object(self):
        if self._some_big_object_from_heavy_io = None:
            self._some_big_object_from_heavy_io = create_some_big_object_from_heavy_io()
        return self._some_big_object_from_heavy_io

shared_objects = SharedObjects()
```

2. 考虑使用进程间共享内存的方式，避免大对象的拷贝带来的内存开销（从python的api来看我们还是需要做序列化的工作）


到这里我们应该对如何优化python程序性能有了一定的认识了。有其他问题欢迎留言讨论交流。
