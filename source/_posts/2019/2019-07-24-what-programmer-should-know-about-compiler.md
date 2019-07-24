---
title: 程序员需要知道的编译器知识
categories:
- compiler
tags:
- compiler
- lex
- yacc
- python
date: 2019-07-24 20:00:00
---

我们每天都在用某种编程语言写代码，但是我们真的了解它们是如何工作的吗？代码是如何转换为计算机可以识别的机器码的呢？

了解编程语言的工作原理，不仅仅可以让我们深入理解某一种编程语言，某些时候还可以帮助以一种更优雅的方式实现想要的功能。比如我们近几年都在谈DSL（Domain Specific Language），在定义了DSL之后，是不是我们可以更进一步的做分析呢？如进行自动错误检查，甚至可以将其转化为某种可以执行的程序？

那么编程语言是如何工作的呢？详细的理论知识得回到我们大学时代的编译原理课程。下面我将主要从实例出发，辅以必备的几个概念介绍来帮助理解编程语言的工作方式，并介绍一些实用的工具帮助我们日常的开发。

## 一个例子

首先我们来看一个最简单的四则运算的例子。在这里我们想定义一种支持简单的 **正整数四则运算** 的语言。

首先是语言中的概念和符号：

- 正整数： `[0-9]+` (正则表达式定义)
- 运算符号： `+` `-` `*` `/`

然后我们定义`*`和`/`的优先级一样，`+`和`-`的优先级一样，`*` `/`高于`+` `-`，与我们一般的理解一致。

这个描述我们可以用一种更规范的形式来表示，如下：

```
tokens:
    NUMBER  : r'[0-9]'

expressions:
    expr    : NUMBER
            | expr + expr
            | expr - expr
            | expr * expr
            | expr / expr
```

`expressions`中的语句采用了递归的形式进行定义（`|`表示或者），这样可以满足任意长度的任意组合了。

像这样一种更规范的格式就是我们所谓的上下文无关文法了，其中著名的BNF范式就是其中一种。

## 几个概念

什么是上下文无关文法？

要理解这个概念，我们先得知道形式语言。形式语言是用 **精确的** 数学或机器可处理的 **公式** 定义的语言。这个概念是专门为计算机定义的，但仅仅从这句话来看，其实还是模糊的，什么样的公式才是数学和机器可处理的呢？

这就牵涉到形式文法的概念，文法用于构成语言，形式文法就是用于构成形式语言，文法可以类比我们自然语言中的*主谓宾*、*主系表*这类结构规则。形式文法可以表示为一个四元组（非终结符N, 终结符Σ, 产生式规则P, 起始符S），从而我们得到了数学上的定义，后面就可以用数学逻辑推导相关文法理论了。到这里大家应该有一点概念了，本文并不想过多涉及理论，详细的数学分析及示例可以参考[wiki](https://zh.wikipedia.org/wiki/%E5%BD%A2%E5%BC%8F%E6%96%87%E6%B3%95)。

上下文无关文法，顾名思义，就是上下文无关的一种形式文法，这种文法虽然简单，但是非常重要，因为所有的编程语言都是基于它来设计的。BNF（Backus Normal Form），也就是巴克斯-诺尔范式（由他们最先引入），就是我们经常用来表达上下文无关文法的一种符号集。

## 理解语言

有了语言定义，我们可以做什么呢？一个直接的任务应该就是理解语言并将其转化为计算机可以执行的代码，以便在机器上运行。完成这个任务的东西我们叫编译器。

很多时候其实我们并不一定要让计算机计算出一个结果，而是只要计算机能理解语言。如何才算理解了语言？对于某一行代码，事实上可以用一棵树结构来描述它，树结构一个节点代表语言定义的一个简单推导。比如`1 + 2 + 3`可以表示成：
```
expr   |-- NUMBER(3)
       +
       |-- expr   |-- NUMBER(1)
                  +
                  |-- NUMBER(2)
```

这里的树结构被称为语法树，有了这颗语法树，我们就可以说理解语言了，因为从这颗语法树我们有办法将其翻译为机器码。

## 理解四则运算语言

理解语言一般我们可以分为两个步骤，第一是理解词，第二是按照语法规则将词组织成语法树。比如四则运算的例子，我们在构造语法树的时候，输入一个字符串，然后需要依次提取字符串中的词（这里的词是指正整数和`+-*/`符号），最后根据词和规则来构造语法树。

在这个简单的例子中，识别词的过程我们可以用简单的正则匹配来实现。但是构造语法树的时候，我们需要构造一个有限状态机（finite-state machine）来实现，这看起来就好像并不是一件简单的事了。

事实上，关于计算机语言的分析早在上世纪50年代就有了比较系统的研究了，相关工具当然也是非常成熟和丰富。最早的工具莫过于`lex(lexical analyser)`和`yacc(yet another compiler-compiler)`了。`lex`就是用来做词法分析的，`yacc`可以用`lex`的词法分析结果来生成语法树。这两个工具可以帮我们生成理解语言的源代码。但是这两个工具是unix系统下的工具，而我们现在用的一般都是GNU的系统。在GNU的系统下的两个类似实现是`flex`和`bison`，ubuntu系统下我们可以用`sudo apt-get install flex bison`来安装。

由于这两个工具是生成c语言的语法分析器代码。为了简单的在python下做一些实验，我们使用[ply](https://github.com/dabeaz/ply)这个工具，它的全称是`python lex-yacc`，也就是python版本的lex和yacc。参考[ply的文档](http://www.dabeaz.com/ply/ply.html)，要实现上述四则运算，我们可以编写代码如下：

```python
tokens = ('NAME', 'NUMBER', )
literals = '+-*/'

t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

precedence = (('left', '+', '-'), ('left', '*', '/'), )

def p_statement_expr(t):
    """statement : expression"""
    print(t[1])

def p_expression_binop(p):
    """expression : expression '+' expression
                  | expression '-' expression
                  | expression '*' expression
                  | expression '/' expression
    """
    if p[2] == '+': p[0] = p[1] + p[3]
    elif p[2] == '-': p[0] = p[1] - p[3]
    elif p[2] == '*': p[0] = p[1] * p[3]
    elif p[2] == '/': p[0] = p[1] / p[3]

def p_expression_number(p):
    """expression : NUMBER"""
    p[0] = p[1]

def test_lex_yacc():
    import ply.lex as lex
    import ply.yacc as yacc
    lexer = lex.lex()
    parser = yacc.yacc()

    while True:
        try:
            s = input('calc > ')
            lexer.input(s)
            while True:
                tok = lexer.token()
                if not tok:
                    break
                print(tok)
            parser.parse(s, lexer=lexer)
        except EOFError:
            break


if __name__ == '__main__':
    test_lex_yacc()
```

运行这个程序，输入我们的表达式，将能得到正确的结果，如下：

```
calc > 1+2*3
LexToken(NUMBER,1,1,0)
LexToken(+,'+',1,1)
LexToken(NUMBER,2,1,2)
LexToken(*,'*',1,3)
LexToken(NUMBER,3,1,4)
7
```

## 一个更复杂的例子

上述的例子看起来过于简单，实际上我们可以用ply实现非常复杂的语法分析器。比如我们可以实现一个用于识别python的函数定义的语法分析器。有了这个分析器，我们就可以从识别结果中获取函数名，参数名，类型信息等，从而完成一些类似代码质量分析，自动代码格式化等工作。

初步看这个问题，似乎正则表达式可以解决一定的问题，但是仅仅有正则表达式是不够的，因为python的很多语法规则过于复杂，难以通过正则表达式来表达。

为识别python的函数定义，我们可以编写测试如下：

```python
def test_yacc_for_python_func_def():
    test_code_lines = [
        'def abc(a,):',
        'def abc(a,):',
        'def abc(a,#xxx()[]\n):',
        'def abc(a: List[int],):',
        'def abc() -> int:',
        'def abc(a, b) -> List[int]:',
        'def abc(a, b: Union[int, List[float]],) -> Union[int, float]:',
        'def abc(a,) -> Union[int, List[float]]:',
        'def abc(a,) -> Union[int, List[float], float]:',
        'def abc(a,) #xxx()[]\n -> Union[int, List[float], float]:'
    ]
    for line in test_code_lines:
        yacc.parse(line, lexer=lexer)
```

有兴趣的小伙伴可以自己尝试实现，或者参考[这里](https://github.com/gmlove/experiments/blob/master/python_lex_yacc/py_func_def.py)我实现的一个版本。

## 更实用的工具

上面的工具已经可以帮我们做很多了，看起来甚至自己定义一门编程语言也是可能的。然而这件事的难度在于，对于一门好用的编程语言，语法定义要足够丰富好用，性能要足够高，相关生态要能做得起来。这就不是单纯的技术活了。

在日常的开发活动中，我们可能接触最多的还是对于当前流行的编程语言的处理。比如，某一天我们可能想要实现一个工具将一个java实现的库，转换为python实现的库，[这里](https://github.com/natural/java2python)就有一个不错的尝试。又比如，某一天我们想要改进我们的IDE，尝试做更多特殊的自动代码格式化支持。还比如，某一天我们想要自动化生成一些代码，就像IDE里面的重构一样。这个时候有没有什么工具可以帮助我们呢？

当然是有的，这里想要提一下 **ANTLR**。代码库在[这里](https://github.com/antlr/antlr4)。这是一个java实现的类似工具，支持的语言非常广泛，我们可以在[这里](https://github.com/antlr/grammars-v4)找到一个列表。可以看到这里支持了go python3 java9等等非常多的语言，基本上我们日常用到的语言都有覆盖了。而且这个工具可以生成各种目标语言的语法分析器。比如我们想得到一个python语言实现的go语言分析器，这个工具可以很容易实现。类似的工具还有很多，我们可以参考wiki上面的一个[比较](https://en.wikipedia.org/wiki/Comparison_of_parser_generators)

如果我们只想用一种编程语言去分析该语言自身，这个时候更简单的方式是直接用语言本身提供的语法分析器。一般提供了JIT(just in time的缩写, 也就是即时编译编译器)功能的语言，都有相应的接口去做语法分析。比如[这里](https://docs.python.org/3/library/ast.html)有python的`ast`库，调用`ast.parse`，输入一段源代码，就得到一颗语法树。javascript有一个第三方库`esprima`([这里](https://esprima.org/))也可以做类似的事情。

到这里我们是不是对日常使用的编程语言有了更深入的了解呢？编译技术是一门强大的技术，灵活运用将能实现很多平时看起来很难的功能。希望当我们遇到这些问题的时候能有一些新的思路。


参考：
- http://dinosaur.compilertools.net/
- https://zh.wikipedia.org/wiki/%E5%B7%B4%E7%A7%91%E6%96%AF%E8%8C%83%E5%BC%8F
- https://zh.wikipedia.org/wiki/%E5%BD%A2%E5%BC%8F%E6%96%87%E6%B3%95
- https://zh.wikipedia.org/wiki/%E4%B8%8A%E4%B8%8B%E6%96%87%E6%97%A0%E5%85%B3%E6%96%87%E6%B3%95
- https://en.wikipedia.org/wiki/Comparison_of_parser_generators

