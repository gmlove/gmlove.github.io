---
title: 程序员需要知道的C/C++编译知识
categories:
- 编译
- c_c++
tags:
- 编译
- 编译器
- GCC
- CL
- CLang
- C
- C++
date: 2020-03-29 20:00:00
---

作为一个非专业`c/c++`开发人员，相信很多人跟我一样，常常会在跟`c/c++`打交道时碰到困难。然而，我们所使用的很多底层的库或软件，却有大量是用`c/c++`编写而成。所以，了解一些基本的`c/c++`知识对于非专业`c/c++`开发人员将非常有帮助。

在下面这些典型的场景中，我们可能会需要用到这些知识：

- 当由于平台需要，我们需要自己编译某些`c/c++`项目
- 当需要在非`c/c++`程序里面进行少量的`c/c++`开发，并与`c/c++`代码交互
- 遇到一些常见的库找不到、版本不兼容等问题

本文尝试总结一下基本的`c/c++`知识，包括常见的平台、静态库/动态库的原理、基础编译指令等。并将结合一些实例来加深理解。

<!-- more -->

## 平台

与一般的跨平台语言（如java、python、nodejs等）不同，如果我们要用`c/c++`来开发一个项目，首先要考虑的问题就是平台支持问题。

什么是平台？一般而言，我们可以将平台理解为一套基础设施，它由一组特定的硬件和软件构成，并使得应用软件可以运行于其上。从硬件层面上讲，平台会主要根据cpu架构不同而不同。由于cpu指令集不同，虽然是同样的代码，往往也会编译为不同的机器码来执行，这就造成了不同平台间显著的差异。从软件层面上讲，平台会主要根据操作系统不同而不同。由于操作系统不同，应用程序接口及系统调用也相应不同，这也造成了不同平台间显著的差异。

一些常见的平台比如：

- Intel 32/64位 CPU + Linux / Windows / macOS
- Arm CPU + Linux / Windows / macOS

如果我们日常每写一行代码都要去考虑平台支持，那将大大降低效率。事实上，现在我们的`c/c++`程序都会基于一些基础的跨平台库来开发。最经典的莫过于标准`c`库和标准`c++`库了，我们开发的几乎所有应用层程序都是基于这些标准库的。

然而这些库的跨平台性怎么样呢？这里不得不提到POSIX标准。

`POSIX` 的全称是 Portable Operating System Interface。它是为维护操作系统间的兼容性而定义的一系列标准。`POSIX`定义了操作系统应用程序接口，`shell`及一些实用工具。最初是为 `Unix` 系列操作系统定义，所以在`Unix`系列操作系统中能拥有良好的兼容性。一些`POSIX`兼容的操作系统包括 `macOS` `Solaris` `AIX`等，还有由华为公司维护的`EulerOS`。拥有绝大部分兼容性的包括 `Android` `GNU/Linux` `OpenBSD` `FreeBSD`等。而 `Windows` 对于`POSIX`标准的兼容性几乎都是由社区提供，如`Cygwin` `MinGW`等，微软自己提供的`C Runtime Library`只实现了常用的接口，兼容性具有不确定性。

既然是这样，我们就多多少少需要关注一下代码的跨平台性了。如果我们只调用常用的标准库`API`，那么程序的兼容性一般是有保障的。而如果我们调用一些平台相关的`API`，那么在向其他平台移植时，将不得不考虑如何处理这些`API`。

一般而言，我们在开发`c/c++`程序时，需要考虑支持大家广泛使用的平台，如 `Intel 64bit CPU` + `Linux` / `Windows` / `macOS` 。这主要是由于我们很可能有人在 `macOS` 或 `Windows` 上面进行日常的开发工作，而程序最终被发布到 `Linux` 上面去运行。

## 编译过程与依赖库

如果我们只需要编写一个比较简单的没有依赖库的应用，我们可能根本不需要关心程序库。现代的编译器或者IDE会自动帮我们处理好内部的库引用问题。但是，一旦我们的程序比较复杂，或者需要引用其他非标准库，我们就需要关心程序库的运行机制了。

一个最简单的 `HelloWorld` 程序可以用`c`语言编写如下：

```c
// hello_world.c
#include<stdio.h>

int main(int argc, char* argv[]) {
    printf("Hello World!\n");
}
```

在`Linux`下，如果我们要将其编译为一个可执行的程序，使用`gcc`编译器，只需要运行命令`gcc hello_world.c`即可。运行此命令之后，`gcc`会在当前目录下生成一个名为`a.out`的可执行程序。运行此程序就可以在控制台打印`Hello World!`了。

看起来整个过程似乎跟程序库没有关系，但是如果我们思考一下`printf`函数是如何来的，就会发现情况不对。其实，就算是这个简单的程序，背后也会有一个程序库来支持，它就是前面提到的`c`标准库。`printf`函数是`c`标准库提供的一个`API`，在`Linux`下面，它的二进制代码一般位于文件`/usr/lib/x86_64-linux-gnu/libc.so`中。

事实上，整个编译过程将分为以下4个步骤完成：

1. 预处理，处理源代码中的文件包含、宏展开等，通过`gcc -E hello_world.c`命令可以看到预处理结果
2. 编译，将预处理后的文件编译为汇编代码，通过`gcc -S hello_world.c`命令可以生成汇编文件`hello_world.s`
3. 汇编，将编译之后的汇编代码生成可重定向的二进制文件，通过`gcc -c hello_world.c`命令可生成文件`hello_world.o`
4. 链接，将可重定向文件与库文件一起链接生成可执行的二进制文件，通过`gcc hello_world.o`可生成文件`a.out`

在`macOS`和`Windows`上，我们可以使用[`llvm clang`](https://clang.llvm.org/docs/UsersManual.html#introduction)和[`cl`](https://docs.microsoft.com/en-us/cpp/build/reference/compiler-options-listed-by-category?view=vs-2019)命令进行编译，编译过程与上述过程类似。

如何查看生成的二进制可执行文件中链接的库呢？

在`Linux`中，我们可以通过`ldd`命令来查看二进制文件中链接的库。如果我们执行`ldd a.out`，即可以看到类似下面的输出：

```
# ldd a.out
    linux-vdso.so.1 (0x00007ffd7d934000)
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f6006bac000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f600719f000)
```

而在`macOS`和`Windows`上，我们可以使用`otool -L a.out`和`dumpbin /dependents hello_world.exe`达到相似的目的。

## 静态库与动态库

上面我们看到了一个简单的单文件源代码程序的编译，那么对于一个多文件源代码程序，情况是怎么样的呢？事实上编译器会将文件一个接一个进行编译，然后再通过第四步将编译好的二进制文件链接成为一个可执行程序。

比如，我们要实现一个乘法运算，有两个源代码文件及一个用于引用的头文件，代码如下：

```c
// mul.c
int mul(int a, int b) {
    return a * b;
}

// mul.h
int mul(int a, int b)

// main_mul.c
#include<stdio.h>
#include "mul.h"
int main(int argc, char* argv[]) {
    printf("mul of %d and %d is %d", 2, 4, mul(2, 4));
}
```

运行`gcc --save-temps mul.c main_mul.c`即可生成可执行程序`a.out`，并保留所有的临时文件。

这里生成的可执行文件在运行时不再需要`mul.o`文件的存在了，它内部其实已经包括了`mul.o`文件的内容。这时，在程序进行链接时，`mul.o`与`main_mul.o`两个文件静态的链接到了一起。

如果我们想独立的发布`mul.c`文件中的内容，作为一个依赖库供其他人使用，该如何操作呢？这里我们就要用到静态库了。我们可以将多个中间二进制文件(`.o`文件)打包为一个文件，然后向他人提供这个文件。

通过命令`ar -rv libmul.a mul.o`即可生成一个名为`libmul.a`的静态库文件。而想要链接这个静态库文件，我们只需要运行命令`gcc -L. -lmul main_mul.c`即可生成与前面相同的`a.out`可执行程序。

用于`macOS`下编译器的`llvm clang`提供了与`gcc`兼容的命令行参数，我们只需要将上述`gcc`更换为`clang`即可达到相同的效果。`Windows`下，我们需要运行`cl mul.c mul.lib`以生成一个静态库文件，然后运行`cl /Femain.exe main_mul.c mul.lib`生成可执行程序。

使用静态库一个不方便的地方在于，库与可执行程序打包到了一起，这会导致生成的可执行程序较大，并且不方便库进行独立升级。这时，聪明的开发者们又想到了其他的办法，那就是动态库。动态库以一个独立的文件形式提供，程序在生成时并不打包动态库的内容，而是在运行时与库进行动态的链接。这就可以解决上面的两个问题了。

如何创建动态库呢？使用`gcc`，我们只需要运行命令`gcc -shared -fPIC mul.c -o libmul.so`即可生成一个名为`libmul.so`的动态库文件。而在创建可执行程序时，需要运行命令`gcc main_mul.c -L. -lmul`。在`macOS`下将`gcc`替换为`clang`即可。在`Windows`下，则运行`cl /LD mul.cc`及`cl main_mul.c /link mul.lib`即可。

大家可能注意到了在`Linux`和`macOS`下都需要在生成的库文件名添加一个`lib`前缀，这是由于历史原因造成的，链接器`ld`在查找库文件时会自动添加此前缀。

还需要注意的一点是，在`Windows`上面直接运行上述命令会失败，因为为了定义一个动态库函数，我们一般需要在函数定义时添加一个`__declspec(dllexport)`编译符号。而在使用动态库函数时，需要在声明函数时，显示的添加前缀`__declspec(dllimport)`。具体的解释，请参考[这里](https://docs.microsoft.com/en-us/cpp/build/exporting-from-a-dll?view=vs-2019)。

## 动态库的工作原理

虽然很多平台都实现了动态库的功能，但是这些实现之间却有所不同。了解了动态库的实现原理，在遇到的动态库相关问题时，我们就可以更从容的去解决。下面对动态库实现原理进行简要介绍。

首先我们了解一下编译出来的二进制文件内容（这里的二进制文件包括动态库文件、中间二进制文件、可执行文件）。各个平台虽然都有自己的二进制格式标准，但大都基于一种通用的[`coff`](https://zh.wikipedia.org/wiki/COFF)(Common Object File Format)格式演进而来。在`linux`下，二进制文件采用`elf`格式，`windows`使用`pe`格式，`macOS`使用`mach-o`格式。虽然有所不同，这些格式都包括这几种元素：

- 用于确定文件类型的魔数（Magic Number，比如`elf`格式为`7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00`）
- 包含文件信息的表（Program header），比如文件运行的平台，编译信息，长度等
- 节头（section headers），二进制文件按节进行组织
- 节体（section data），具体的节的内容

对于同样的源代码文件在不同的平台编译，如果`cpu`相同，那么编译出来的二进制机器码也几乎是相同的。它们之间的差异通常在这几方面：

- 链接的`c`库，不同的平台有不同的`c`库实现
- 启动和退出逻辑
- 节组织，`elf`格式的代码段一般是`.text`，而`pe`格式为`.code`

除机器码之外，二进制文件的其他节的内容是为密切配合操作系统的二进制文件加载方式而设计实现的。

源代码经过编译得到中间二进制文件，但是由于每个源代码文件单独编译，它们并不知道自己引用的外部函数或变量的地址。编译时通常将这些外部符号地址设置为一些特殊值，并记录到特定的节中，以便链接时可以正确的对他们进行修正。比如

如果我们用`objdump -S main_mul.o`命令查看前面编译出来的文件的汇编代码，可以看到以下汇编代码：

```
main_mul.o:     file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <main>:
   0:   55                      push   %rbp
   1:   48 89 e5                mov    %rsp,%rbp
   4:   48 83 ec 10             sub    $0x10,%rsp
   8:   89 7d fc                mov    %edi,-0x4(%rbp)
   b:   48 89 75 f0             mov    %rsi,-0x10(%rbp)
   f:   be 04 00 00 00          mov    $0x4,%esi
  14:   bf 02 00 00 00          mov    $0x2,%edi
  19:   e8 00 00 00 00          callq  1e <main+0x1e>
  1e:   89 c1                   mov    %eax,%ecx
  20:   ba 04 00 00 00          mov    $0x4,%edx
  25:   be 02 00 00 00          mov    $0x2,%esi
  2a:   48 8d 3d 00 00 00 00    lea    0x0(%rip),%rdi        # 31 <main+0x31>
  31:   b8 00 00 00 00          mov    $0x0,%eax
  36:   e8 00 00 00 00          callq  3b <main+0x3b>
  3b:   b8 00 00 00 00          mov    $0x0,%eax
  40:   c9                      leaveq
  41:   c3                      retq
```

其中第`0x19`位置的指令使用`callq`调用了`mul`函数，这里的`mul`函数的地址是`0x1e`，对应重定位代码节`.rela.text`中的`mul`。使用命令`readelf -r main_mul.o`可以看到重定位代码节的内容：

```
Relocation section '.rela.text' at offset 0x270 contains 3 entries:
  Offset          Info           Type           Sym. Value    Sym. Name + Addend
00000000001a  000b00000004 R_X86_64_PLT32    0000000000000000 mul - 4
00000000002d  000500000002 R_X86_64_PC32     0000000000000000 .rodata - 4
000000000037  000c00000004 R_X86_64_PLT32    0000000000000000 printf - 4

...
```

而当我们用`objdump -S a.out`命令查看链接得到的可执行文件时，可以看到链接器对于位置修正的结果：

```
000000000000064a <main>:
 64a:   55                      push   %rbp
 64b:   48 89 e5                mov    %rsp,%rbp
 64e:   48 83 ec 10             sub    $0x10,%rsp
 652:   89 7d fc                mov    %edi,-0x4(%rbp)
 655:   48 89 75 f0             mov    %rsi,-0x10(%rbp)
 659:   be 04 00 00 00          mov    $0x4,%esi
 65e:   bf 02 00 00 00          mov    $0x2,%edi
 663:   e8 92 00 00 00          callq  6fa <mul>
 668:   89 c1                   mov    %eax,%ecx
 66a:   ba 04 00 00 00          mov    $0x4,%edx
 66f:   be 02 00 00 00          mov    $0x2,%esi
 674:   48 8d 3d 19 01 00 00    lea    0x119(%rip),%rdi        # 794 <_IO_stdin_used+0x4>
 67b:   b8 00 00 00 00          mov    $0x0,%eax
 680:   e8 9b fe ff ff          callq  520 <printf@plt>
 685:   b8 00 00 00 00          mov    $0x0,%eax
 68a:   c9                      leaveq
 68b:   c3                      retq

...

00000000000006fa <mul>:
 6fa:   55                      push   %rbp
 6fb:   48 89 e5                mov    %rsp,%rbp
 6fe:   89 7d fc                mov    %edi,-0x4(%rbp)
 701:   89 75 f8                mov    %esi,-0x8(%rbp)
 704:   8b 45 fc                mov    -0x4(%rbp),%eax
 707:   0f af 45 f8             imul   -0x8(%rbp),%eax
 70b:   5d                      pop    %rbp
 70c:   c3                      retq
 70d:   0f 1f 00                nopl   (%rax)

...
```

上面`0x663`位置的指令对应前面`main_mul.o`中的`0x19`位置的指令，而这里的`mul`函数的地址已经被修正为`0x6fa`，即函数`mul`第一条指令的地址。

如果是静态链接，最后得到的是一个大的二进制文件，里面的符号地址可以在链接时全部被正确修正。但是如果是动态链接，情况就比较复杂了。由于操作系统将动态库加载到什么地址会动态变化，是不确定的，所以也就不能简单的预先进行地址修正。

事实上，之所以称作动态链接，正是由于这些库的链接过程（地址修正过程）是在运行时完成的。关于动态链接的原理，可以简单说明如下。对于一个动态库，一般情况下，我们首先将其编译为一个地址无关代码存储起来（地址无关代码可以简单理解为用相对地址进行变量或函数寻址，这也是`gcc`编译时参数`-fPIC`的作用，`PIC`的全称就是position independent code），当操作系统在加载这些地址无关代码时，动态链接程序会记录加载之后得到的变量或函数的真正地址到一个映射表(`GOT`)中，供使用库的进程查询。其次，在链接可执行文件时，编译器会将所需要链接的动态库及其版本写入到二进制文件的某些节中，这样，在程序运行时就可以根据这些信息去查询到相应的库函数了。

事实上，经过编译链接的可执行文件并不是一开始就执行我们定义的`main`函数，而是会执行`c`库中的一些启动函数。对于`linux` `glibc`而言，这个函数是`glibc`中的`_start`函数，代码可以参考[这里](https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/start.S)。这是一个用汇编语言编写的函数，它会进一步调用[`libc-start.c`](https://github.com/lattera/glibc/blob/master/csu/libc-start.c)中的`__libc_start_main`函数完成启动工作。对于动态链接的程序，在`c`库中的启动函数会调用链接器函数进行一定的初始化工作，包括动态库的查找，加载，初始化等。

到这里，我们应该大致了解了可执行程序及库的加载和运行机制。


## 常见问题

### 在`Linux`下遇到`glibc`版本不同

可以使用工具[`patchelf`](https://www.mankier.com/1/patchelf)修改二进制文件，对链接的库进行修正，但是这样就需要我们自己去保证库的版本兼容性了。一个典型的修正链接库路径的命令如下：

`./patchelf --set-interpreter /path/to/newglibc/ld-linux.so.2 --set-rpath /path/to/newglibc/ myapp`

### 在`windows`下遇到`dll`文件找不到

我们可以使用工具[`dependency walker`](https://www.dependencywalker.com/)找出程序的所有依赖库，并识别系统中没有的库文件。这些找不到的库文件一般都是`windows`的开发工具`Visual Studio`提供的库文件。有些程序没有在安装程序中提供这些动态库文件的拷贝，而是默认用户的系统中已经存在这些库了，这就造成`dll`文件找不到的问题。

参考：

- https://eli.thegreenplace.net/2011/08/25/load-time-relocation-of-shared-libraries/
- https://eli.thegreenplace.net/2011/11/03/position-independent-code-pic-in-shared-libraries/
- https://www.cnblogs.com/catch/p/3857964.html
- https://stackoverflow.com/questions/847179/multiple-glibc-libraries-on-a-single-host









