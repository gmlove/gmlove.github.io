---
title: Hadoop安全认证机制 （二）
categories:
- 大数据
- 安全
tags:
- 大数据
- 安全
- hadoop
- kerberos
date: 2019-10-30 20:00:00
---

在[上一篇](http://brightliao.me/2019/10/27/hadoop-auth/)文章中，我们分析了`Kerberos`协议的设计和通信过程。可以了解到，`Kerberos`主要实现了不在网络传输密码的同时又能在本地进行高性能鉴权。由于`kerberos`的协议设计相对复杂，看到评论有人还有疑问，这里再举一个例子来分析一下`kerberos`的安全性。

## `Kerberos`协议回顾

假设有三个组件A B C，A想和C进行安全通信，而B作为一个认证中心保存了认证信息。那么以以下的方式进行通信就可以做到安全：

1. A向B请求说要访问C，将此消息用A的秘钥加密之后，发给B
2. B验证A的权限之后，用A自己的秘钥加密一个会话密码，然后传给A
3. 同时B还向A发送一个A自己不能解密，只能由C解密的消息
4. A在解密会话密码之后，将需要和C通信的消息（业务消息）用这个会话密码加密然后发给C，同时A需要将B发给A而A又不能解密的消息发给C
5. C在拿到消息之后，可以将第三步中的消息解密，得到会话秘钥，从而可以解密A发过来的业务消息了

<!-- more -->

整个过程，A无需知道C的密码，C也无需知道A的密码就可以完成安全通信。

这里的安全性我们可以从以下几个方面来看：

1. 如果消息被截获
当第一步中的消息被截获：这里的消息用A的秘钥加密了，截获也无法解密
当第二步中的消息被截获：这里的消息用A的秘钥加密了，截获也无法解密
当第三步中的消息被截获：这里的消息用C的秘钥加密了，截获也无法解密
当第四步中的消息被截获：这里的消息分别用会话秘钥、C的秘钥加密了，截获也无法解密
当第五步中的消息被截获：这里的消息用会话秘钥，截获也无法解密

2. 如果A是一个攻击方（某一个有权限的用户想要提权）
他只能拿到自己的秘钥，而无法获取B或C的秘钥，他不能随意生成一个加密消息发给C请求服务（冒充其他用户），因为他无法伪造有会话密码而又用C的秘钥加密的消息

3. 如果C是一个攻击方（欺骗某个有权限的用户）
他无法解密第三步中的消息，所以无法解密A的消息，从而也就无从提供服务

4. 如果B是一个攻击方
他无法解密A的消息，从而无法提供服务

5. 如果传输的消息被破解（任何加密都是可以被破解的，只是时间的问题）
由于整个通信过程由会话秘钥来加密，会话秘钥的有效期通常比较短，当消息被破解之后，攻击者也不能利用破解得到的秘钥去破解后续的消息

从这几个方面来看，这个协议都是比较安全的。

以上的安全通信步骤是`kerberos`安全的核心机制，A对应文章中的`Client`，B对应文章中的`TGS`，C对应文章中的`SS`。

但`kerberos`还引入了一个AS的组件，这主要为了提高性能和扩展性。

有了`AS`之后，我们可以将整个通信看成两个上述ABC通信模式的重复。第一个通信模式A对应文章中的`Client`，B对应文章中的`AS`，C对应文章中的`TGS`，为了实现`Client`和`TGS`的安全通信。第二个通信模式A对应文章中的`Client`，B对应文章中的`TGS`，C对应文章中的`SS`，为了实现`Client`和`SS`的安全通信。

为什么有了两次通信模式之后，就能提高性能和扩展性呢？实际上一般我们可以将`Client/TGS`的会话秘钥有效期配置得更长一些，而将`Client/SS`的会话秘钥有效期配置得比较短。由于一旦我们有一个有效的`TGT`及`Client/TGS`会话秘钥，在这个秘钥的有效期内，我们无需再访问`AS`去生成新的会话秘钥。当`Client/TGS`会话秘钥有效期较长的时候，我们就可以较少的访问`AS`，从而将`AS`这一第一入口服务的负载降低。而`TGS`由于需要经常参与秘钥生成，它的负载会相对较高，这里我们就可以将`TGS`扩展到多台服务器来支撑大的负载。`AS`可以给`Client`提供一个有效的`TGS`地址，从而实现`TGS`的分布式扩展。

## `Kerberos`协议发展

### `GSS API`

`Kerberos`协议本身只是提供了一种安全认证和通信的手段，要应用这个协议，我们需要一套`API`接口。在具体实现的时候，每个人都会写出不一样的代码，从而产生不同的`API`。这可不是好事，对于应用方而言，不仅仅学习成本高，而且系统迁移能力差，比如换一个`Kerberos`服务器可能就会出现兼容性问题。就像`windows`上面的换行用`\r\n`，而`unix`类操作系统用`\n`，这给每一个开发者都带来了麻烦。

所以，在具体的工程应用时，一种通用的`API`就变得非常重要。这就是`GSS API`，其全称是The Generic Security Services Application Program Interface，即通用安全服务应用程序接口。这套`API`在设计的时候其实不仅仅考虑了对于`Kerberos`的支持，还考虑了支持其他的协议，所以称为通用接口。由于我们总是会发展出其他的安全协议的，抽象一套可以长期保持不变的通用的`API`接口，就可以避免应用层进行修改。这一套`API`接口就是在上一篇文章中我们用到的接口了。

从`GSS API`接口来看，我们的认证过程可以抽象为这样几个简单的步骤：

- 客户端：创建一个`Context`上下文用来保存数据 -> 通过`initSecContext`获取一个`token` -> 将`token`发送给服务器 -> 等待服务器回发的用于通信的`token`
- 服务器：创建一个`Context`上下文用来保存数据 -> 读取客户端发来的`token` -> 验证`token`，并（可能）生成一个新的用于通信的`token` -> 将`token`发给客户端

这里的认证过程简单到甚至没有出现认证服务器，基于这样的一套通用`API`去实现其他应用就相对轻松多了。`Kerberos`内部的通信细节，多次传输的各种密文全部都隐藏在这样的`API`实现中。具体的`GSS API`使用代码示例，大家可以参考上一篇文章中代码。

### SPNEGO

由于`GSS API`设计可以支持多种安全协议，另一个想法会自然的冒出来。我们可以让服务器支持多种认证协议，然后具体用哪种，由客户端和服务器端协商决定。这就使得我们在开发应用时可以给最终的用户提供选择，便于使用他或她所偏好使用的认证方式，从而带来更好的用户体验。同时，服务器和客户端在各自实现时，也可以相互独立的增量式的添加或去掉对于某一具体协议的支持，而不用完全同步的进行修改。这对于同一个服务器要支持多个版本的客户端而言会很有用。

这就是`SPNEGO`了，其全称是Simple and Protected GSSAPI Negotiation Mechanism，即基于`GSS API`实现的一套简单的协议协商机制。这一协议由微软最早提出并应用在windows操作系统中，与我们最贴近的应用，当属于浏览器的系统集成认证了。大家回忆一下我们使用IE浏览器的体验，可以发现，很多网站可以直接使用系统的域账户登录。这就是用`SPNEGO`协议实现的浏览器系统集成认证。在企业中，如果我们为所有员工配置了`windows`域账户，而当我们有一些基于web的企业应用需要认证时，就可以利用这一机制实现无感知的认证。其实不只是IE浏览器，`Firefox` `Chrome`等主流浏览器基本上都实现了这样的系统集成登录机制。

这个协议的通信过程大致为：

1. `Client`向`Server`请求服务
2. `Server`检查`Client`是否有提供有效的认证信息：如果没有，返回消息（包括服务器支持的认证方式）给`Client`，以便`Client`可以完成认证；如果有，就提供服务
3. `Client`完成认证之后，向`Server`请求服务，并带上认证信息
4. 回到第二步中进行认证检查，直到通过或认证次数达到阈值为止

## Hadoop 认证机制

介绍了这么多，其实都是为了我们分析`Hadoop`的认证机制实现。到这里，相信大家应该也猜到了，在`Hadoop`的认证中，各个节点的通信实际上使用的就是`GSS API`去实现的基于`Kerberos`协议的单点认证。而`Hadoop`对外提供的很多基于web的应用，比如Web HDFS、统计信息页面、Yarn Application管理等等，其认证都是基于`SPNEGO`协议的。这两个协议的配置其实在我们后续配置`Hadoop`认证时也是最主要的配置了。

## 相关源代码分析

（下面的内容请大家结合源代码一起分析，仅仅读文字可能有很多内容会难以理解）

### GSS API中的`Kerberos`实现

我们打开`OpenJDK`的[源代码库](https://github.com/unofficial-openjdk/openjdk/blob/5b0f4d762e/src/java.security.jgss/share/classes/module-info.java)，浏览到下面这里的代码：

![OpenJDK](/attaches/2019/2019-10-30-hadoop-auth-2/openjdk.png)

这里的代码量还是挺大的，细节很多，我们一起看一下主要的设计。`GSS API`在Java语言中通过`jgss`模块来实现。`jgss`首先定义了一些底层认证机制需要实现的接口，即`sun.security.jgss.spi`包中的基本接口`GSSContextSpi` `GSSNameSpi` `GSSCredentialSpi`和工厂接口`MechanismFactory`。底层的协议只需要实现这几个接口就行了，关于`Kerberos`的实现在包`sun.security.jgss.krb5`中，其实这个包里面的代码只是对接了真正的`Kerberos`通信协议实现和`GSS API`接口。这里的设计，按照DDD的思想，我们可以理解为一套防腐层，`GSS API`和`Kerberos`可以看成两个独立的领域，通过引入防腐层，它们就可以相互独立的各自演进。当接口有改变的时候，我们只需要修改防腐层的代码就行了。

真正的`Kerberos`协议实现在包`sun.security.krb5`下面，这里的实现通过`javax.security.auth.kerberos`包下面的类对应用层暴露接口（应用层在使用`GSS API`时，有时还是需要关心底层认证机制的相关信息的）。作为应用层，如果有必要获取底层认证机制相关的信息，我们将只使用`javax.security.auth.kerberos`中定义的接口，而无需关心`sun`包下面的实现。这里的实现的核心代码在`Credentials`类中，我们看到其定义了`acquireTGTFromCache` `acquireDefaultCreds` `acquireServiceCreds`等接口用于交换秘钥。更细节的实现代码，大家如有兴趣，可以结合上一篇文章中的通信流程自行研究。我们这里只简要分析一下主要的设计思想。

### `Hadoop`中使用`GSS API`进行认证

`Hadoop`中和认证相关的模块主要有两个：一个是直接使用`GSS API`进行认证，用于`tcp`通信的`org.apache.hadoop.security.UserGroupInformation`类；另一个是基于`SPNEGO`协议进行认证，用于`HTTP`通信的`org.apache.hadoop.security.authentication.server.KerberosAuthenticationHandler`。

`UserGroupInformation`主要用于`Hadoop`各个内部模块间的通信，也可以用于某一个客户端和`Hadoop`的某个模块进行通信，它同时为服务器和客户端的认证提供了支持。比如`NameNode`的启动之后，它将发起一个登陆请求，用于验证给自己配置的`Principal`和`keytab`是否有效（[这里](https://github.com/apache/hadoop/blob/release-2.7.7-RC0/hadoop-hdfs-project/hadoop-hdfs/src/main/java/org/apache/hadoop/hdfs/server/namenode/ha/BootstrapStandby.java)108行）。同时当有内部服务（如某个datanode）的rpc请求到来的时候，它将使用登陆得到的认证主体`Subject`中的`doAs`方法来验证发送过来的认证信息，并进行权限验证。有客户端的rpc请求到来时，它将获取客户端的用户信息，并根据配置的ACL（访问控制列表）进行权限验证（实现见[这里](https://github.com/apache/hadoop/blob/release-2.7.7-RC0/hadoop-hdfs-project/hadoop-hdfs/src/main/java/org/apache/hadoop/hdfs/server/datanode/DataXceiver.java)的1287行，及[这里](https://github.com/apache/hadoop/blob/release-2.7.7-RC0/hadoop-hdfs-project/hadoop-hdfs/src/main/java/org/apache/hadoop/hdfs/security/token/block/BlockPoolTokenSecretManager.java)）。为了缓存认证信息，避免没必要的重新认证，程序需要维护当前登录的账号的信息，这也就是为什么`UserGroupInformation`在设计上定义了很多静态的属性。同时我们可以注意到很多`synchronized` 关键字附加到了某些静态方法上，这是为了支持多线程访问这些全局缓存的信息。

`KerberosAuthenticationHandler`的实现是为了支持在HTTP服务中进行`Kerberos`认证，这个类最终会封装为一个Web服务器中的`Filter`实现对所有HTTP请求的权限验证（[这里的AuthFilter](https://github.com/apache/hadoop/blob/release-2.7.7-RC0/hadoop-hdfs-project/hadoop-hdfs/src/main/java/org/apache/hadoop/hdfs/web/AuthFilter.java)及其[基类AuthenticationFilter](https://github.com/apache/hadoop/blob/release-2.7.7-RC0/hadoop-common-project/hadoop-auth/src/main/java/org/apache/hadoop/security/authentication/server/AuthenticationFilter.java)）。由于基于Servlet的Web服务器有很成熟的接口设计，这个模块的实现也相对独立和简单。可以看到它在`init`的时候使用`GSS API`完成了登录，在`authenticate`的时候，将判断是否有有效的认证信息，如果没有将返回协商认证的HTTP头部消息以便客户端去完成认证，如果有将进行认证并提供服务。

## Web服务器认证实现示例

对于一个运行于Hadoop集群的`Spark`应用，我们通常是通过`spark-submit`命令行工具来向集群提交任务的。这一机制对于`spark`应用的开发者看起来很灵活，但如果我们想进行更多的统一管理，比如限制资源使用、提升易用性等等，这样的机制就略显不足了。这个时候一般的做法是将运行`spark`应用的这一能力封装为一个服务，以便进行统一的管理。`Livy`就是为实现这样的功能而开发的一个开源工具。

使用`Livy`，我们可以使用REST的接口向集群提交`spark`任务。在这里`Livy`其实相当于是整个`Hadoop`大数据集群的一个扩展服务。`Livy`在实现的时候如何进行权限的支持呢？当我们去查看`Livy`的源代码的时候，我们会发现，要为每个请求添加`Kerberos`认证，几乎只需要一行代码，[这里](https://github.com/apache/incubator-livy/blob/v0.5.0-incubating/server/src/main/scala/org/apache/livy/server/LivyServer.scala)的237行即为那行关键的代码。这里Livy就是有效的利用了上面的`KerberosAuthenticationHandler`进行实现的。


## 总结

在简要分析了`Kerberos`的协议和发展及相关的实现代码之后，大家是不是对于这个协议及其在大数据上面的应用有了更深入的理解呢？我想大家一定对于搭建一套支持`Kerberos`认证的安全的大数据集群很有兴趣，后续的文章将与大家一起来尝试搭建一套这样的集群。由于`Hadoop`的安全机制的复杂性，我们在初次搭建这样的集群时可能会碰到各种各样的问题，上面的这些基础知识将为解决这些问题将提供很大的帮助。







