---
title: Hadoop安全认证机制 (一)
categories:
- 大数据
- 安全
tags:
- 大数据
- 安全
- hadoop
- kerberos
date: 2019-10-27 20:00:00
---

安全无小事，我们常常要为了预防安全问题而付出大量的代价。虽然小区楼道里面的灭火器、消防栓常年没人用，但是我们还是要准备着。我们之所以愿意为了这些小概率事件而付出巨大的成本，是因为安全问题一旦发生，很多时候我们将无法承担它带来的后果。

在软件行业，安全问题尤其突出，因为无法预料的事情实在太多了。软件的复杂性让我们几乎无法完全扫清安全问题，模块A独立运行可能没问题，但是一旦和模块B一起工作也许就产生了安全问题。

不可否认为了让软件更安全，我们引入了很多复杂的机制。不少人开发者也抱怨为了进行安全处理而做了太多额外的事情。在一个复杂的分布式软件Hadoop中，我们为此付出的成本将更大。比如，我们可能可以比较轻松的搭建一个无安全机制的集群，但是一旦需要支持安全机制的时候，我们可能会付出额外几倍的时间来进行各种复杂的配置和调试。

Hadoop在开始的几个版本中其实并没有安全机制的支持，后来Yahoo在大规模应用Hadoop之后，安全问题也就日益明显起来。大家都在一个平台上面进行操作是很容易引起安全问题的，比如一个人把另一个人的数据删除了，一个人把另一个人正在运行的任务给停掉了，等等。在当今的企业应用里面，一旦我们的数据开始上规模之后，安全机制的引入几乎是必然的选择。所以作为大数据领域的开发者，理解Hadoop的安全机制就显得非常重要。

Hadoop的安全机制现在已经比较成熟，网上关于它的介绍也很多，但相对较零散，下面我将尝试更系统的，并结合实例代码，给大家分享一下最近一段时间关于Hadoop安全机制的学习所得，抛个砖。

预计将包括这样几个方面：

1. Kerberos协议介绍及实践
2. Kerberos协议发展及Hadoop相关源码分析
3. Hadoop安全集群搭建及测试
4. 周边工具的安全支持

<!-- more -->

## 安全认证协议

### Kerberos

做Web开发的同学们可能比较熟悉的认证机制是`JWT`，近两年`JWT`的流行几乎让其成为了实现单点登录的一个标准。`JWT`将认证服务器认证后得到的`token`及一定的用户信息经过`base64`编码之后放到HTTP头中发送给服务器端，得益于`token`的加密机制（一般是非对称加密），服务器端可以在不连接认证服务器就进行`token`验证（第一次验证时会向认证服务器请求公钥），从而实现高性能的鉴权。这里的`token`虽然看起来不可读，实际上我们经过简单的解码就能得到`token`的内容。所以`JWT`一般是要结合`HTTPS`一起应用才能带来不错的安全性。

![JWT认证机制](/attaches/2019/2019-10-27-hadoop-auth/jwt.png)

`JWT`看起来还不错呀，安全模型比较简单，能不能直接用在`Hadoop`上面呢？可能可以。但是由于`Hadoop`的出现早于`JWT`太多，所以当时的设计者们是不可能考虑使用`JWT`的。实际上`JWT`主要是针对web的场景设计的，对于分布式场景中，很多问题它是没有给出答案的。一些典型的场景比如服务间的认证该如何实现，如何支持其他的协议，等等。`Hadoop`的安全认证使用的是`Kerberos`机制。相比`JWT`，`Kerberos`是一个更为完整的认证协议，然而也正是因为其设计可以支持众多的功能，也给其理解和使用带来了困难。

这里之所以提到`JWT`，是因为`JWT`实际上可以看成是`Kerberos`协议的一个极简版本。`JWT`实现了一部分`Kerberos`的功能。如果我们能对于`JWT`的认证机制比较熟悉，那么对于`Kerberos`机制的理解应当是有较大帮助的。

`Kerberos`协议诞生于MIT大学，早在上世纪80年代就被设计出来了，然后经过了多次版本演进才到了现在我们用的V5版本。作为一个久经考验的安全协议，`Kerberos`的使用其实是非常广泛的，比如`Windows`操作系统的认证就是基于`Kerberos`的，而`Mac` `Red Hat Enterprise Linux`也都对于`Kerberos`有完善的支持。各种编程语言也都有内置的实现。对于这样一个重要的安全协议，就算我们不从事大数据相关的开发，也值得好好学习一下。

`Kerberos`设计的有几个大的原则:
1. 利用公开的加密算法实现
2. 密码尽量不在网络上传输
3. 高安全性和性能
4. 支持广泛的安全场景，如防止窃听、防止重放攻击、保护数据完整性等

那么这个协议是如何工作的呢？与`JWT`类似，`Kerberos`同样定义了一个中心化的认证服务器，不过对于这个认证服务器，`Kerberos`按照功能进一步将其拆分为了三个组件：认证服务器（Authentication Server，AS）、密钥分发中心（Key Distribution Center，KDC）、票据授权服务器（Ticket Granting Server，TGS）。在整个工作流程中，还有两个参与者：客户端(Client)和服务提供端(Service Server，SS)。

`Kerberos`大体上的认证过程与`JWT`一致：第一步是客户端从认证服务器拿到`token`（这里的术语是`Ticket`，下文将不区分这两个词，请根据上下文理解）；第二步是将这个`token`发往服务提供端去请求相应的服务。

下图是整个认证过程中各个组件按顺序相互传递的消息内容，在阅读整个流程之前，有几点提需要注意:
1. 各个组件都有自己独立的秘钥：Client的秘钥由用户提供，AS、TGS、SS需要提前生成自己独立的秘钥
2. AS、TGS由于属于认证服务器的一部分，它们可以查询KDC得到用户或其他服务器的秘钥，比如AS可以认为拥有用户的、TGS的以及SS的秘钥

![Kerberos认证流程](/attaches/2019/2019-10-27-hadoop-auth/kerberos-auth-sequence.png)

看了这个复杂的流程，大家心里应该有很多疑惑。整个通信过程传递了很多的消息，消息被来来回回加密了很多次，真的是有必要的吗？背后的原因是什么呢？事实上，我们结合上面提到的几个设计原则来看，问题就会相对清晰一些。

虽然整个通信过程涉及到的消息很多，但是我们仔细思考就可以发现这几条规律：
1. 整个认证过程中，避免了任何地方有明文的密码传输
2. 与`JWT`一样，通信过程生成有效时间比较短的会话秘钥用于通信
3. 与`JWT`一样，认证服务器无需存储会话秘钥，各个参与方（Client/SS）可以独立进行消息验证，从而实现高性能。这也是虽然消息B和E不能被Client解密，但是还是会发往Client，然后再由Client回发的原因
4. `Kerberos`并没有对`Client`和`SS`之间的通信协议进行限制，虽然和认证服务器进行通信需要基于`TCP/UDP`，但`Client`和`SS`通信可以用任意协议进行

理解了上述通信流程之后，可以看到，相比`JWT`，`Kerberos`还进行了下面的额外验证：
1. 认证过程将验证服务提供端的ID，一般会基于hostname进行
2. 认证过程将验证各个组件的时间，相互不能相差太多，这也是`Kerberos`要求各个组件进行时间同步的原因

除了上面这些安全验证，其实`Kerberos`还支持免密码输入的登录，我们可以将用户的秘钥（并非真正的密码，由真正的密码hash生成）生成到一个`keytab`格式的文件中，这样在第一步中，就可以由用户提供ID(principal)及`keytab`文件来完成了。

虽然`Kerberos`可以支持多种场景的认证，但是由于其协议设计比较复杂，在使用上会给我们带来不少的困难。比如我们需要提前为各个组件生成独立的秘钥，一般要求每个服务器都不一样，与不同的主机绑定，这就给我们部署服务带来了挑战，特别是在当前微服务、云原生应用、容器、k8s比较流行的时候。

### 通信过程演示

为了更清晰的看到整个通信的过程，我们可以动手实践一下看看：

运行下面的命令进入一个`centos`的容器：

```bash
docker run -it centos:7 -p1800:1800 -p1802:1802 bash
```
然后安装配置kdc并生成相关的秘钥：

```bash
# 将kdc kdc.hadoop.com加入hosts，以便后续进行基于hosts文件的主机名解析
yum install net-tools -y
ip_addr=$(ifconfig eth0 | grep inet | awk '{print $2}')
echo "$ip_addr kdc-server kdc-server.hadoop.com" >> /etc/hosts

# 安装相关软件并进行配置
yum install krb5-server krb5-libs krb5-workstation -y
# 创建krb5配置文件，详细配置解释请参考：https://web.mit.edu/kerberos/krb5-1.12/doc/admin/conf_files/krb5_conf.html
cat > /etc/krb5.conf <<EOF
#Configuration snippets may be placed in this directory as well
includedir /etc/krb5.conf.d/

[logging]
  default = FILE:/var/log/krb5.log
  kdc = FILE:/var/log/krb5kdc.log
  admin_server = FILE:/var/log/kadmind.log

[libdefaults]
  forcetcp = true
  default_realm = HADOOP.COM
  dns_lookup_realm = false
  dns_lookup_kdc = false
  ticket_lifetime = 24h
  renew_lifetime = 7d
  forwardable = true
  udp_preference_limit = 1
  default_tkt_enctypes = des-cbc-md5 des-cbc-crc des3-cbc-sha1
  default_tgs_enctypes = des-cbc-md5 des-cbc-crc des3-cbc-sha1
  permitted_enctypes = des-cbc-md5 des-cbc-crc des3-cbc-sha1

[realms]
  HADOOP.COM = {
    kdc = kdc-server.hadoop.com:2802
    admin_server = kdc-server.hadoop.com:2801
    default_domain = hadoop.com
  }

[domain_realm]
  .hadoop.com = HADOOP.COM
  hadoop.com = HADOOP.COM
EOF
# 创建kdc配置文件，详细配置解释请参考：https://web.mit.edu/kerberos/krb5-1.12/doc/admin/conf_files/kdc_conf.html
cat > /var/kerberos/krb5kdc/kdc.conf <<EOF
default_realm = HADOOP.COM

[kdcdefaults]
 kdc_ports = 0
 v4_mode = nopreauth

[realms]
 HADOOP.COM = {
    kdc_ports = 2800
    kdc_tcp_ports = 2802
    admin_keytab = /etc/kadm5.keytab
    database_name = /var/kerberos/krb5kdc/principal
    acl_file = /var/kerberos/krb5kdc/kadm5.acl
    key_stash_file = /var/kerberos/krb5kdc/stash
    max_life = 10h 0m 0s
    max_renewable_life = 7d 0h 0m 0s
    master_key_type = des3-hmac-sha1
    supported_enctypes = arcfour-hmac:normal des3-hmac-sha1:normal des-cbc-crc:normal des:normal des:v4 des:norealm des:onlyrealm des:afs3
    default_principal_flags = +preauth
}
EOF

echo -e '123456\n123456' | kdb5_util create -r HADOOP.COM -s  # 创建一个名为HADOOP.COM的域
/usr/sbin/krb5kdc && /usr/sbin/kadmind                        # 启动kdc及kadmind服务

echo -e '123456\n123456' | kadmin.local addprinc gml    # 创建gml账号
kadmin.local xst -k gml.keytab gml@HADOOP.COM           # 生成gml账号的keytab文件

kadmin.local addprinc -randkey root/localhost@HADOOP.COM       # 创建名为root并和kdc主机进行绑定的服务账号
kadmin.local xst -k server.keytab root/localhost@HADOOP.COM    # 创建用于服务器的keytab文件
```

将生成的keytab文件下载到本地，然后就可以进行测试了。编写测试的客户端和服务端代码如下：

```java
import org.ietf.jgss.GSSContext;
import org.ietf.jgss.GSSCredential;
import org.ietf.jgss.GSSException;
import org.ietf.jgss.GSSManager;
import org.ietf.jgss.Oid;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;


public class Test {

    public static class TestClient {
        private String srvPrincal;
        private String srvIP;
        private int srvPort;
        private Socket socket;
        private DataInputStream inStream;
        private DataOutputStream outStream;

        public TestClient(String srvPrincal, String srvIp, int srvPort) throws Exception {
            this.srvPrincal = srvPrincal;
            this.srvIP = srvIp;
            this.srvPort = srvPort;
            this.initSocket();
            this.initKerberos();
        }

        private void initSocket() throws IOException {
            this.socket = new Socket(srvIP, srvPort);
            this.inStream = new DataInputStream(socket.getInputStream());
            this.outStream = new DataOutputStream(socket.getOutputStream());
            System.out.println("Connected to server: " + this.socket.getInetAddress());
        }

        private void initKerberos() throws Exception {
            System.setProperty("java.security.krb5.conf", "experiment/src/main/krb5.conf");
            System.setProperty("java.security.auth.login.config", "experiment/src/main/client.conf");
            System.setProperty("javax.security.auth.useSubjectCredsOnly", "false");
            System.setProperty("sun.security.krb5.debug", "true");

            System.out.println("init kerberos: set up objects as configured");
            GSSManager manager = GSSManager.getInstance();
            Oid krb5Oid = new Oid("1.2.840.113554.1.2.2");
            GSSContext context = manager.createContext(
                    manager.createName(srvPrincal, null),
                    krb5Oid, null, GSSContext.DEFAULT_LIFETIME);
            context.requestMutualAuth(true);
            context.requestConf(true);
            context.requestInteg(true);

            System.out.println("init kerberos: Do the context establishment loop");

            byte[] token = new byte[0];

            while (!context.isEstablished()) {
                // token is ignored on the first call
                token = context.initSecContext(token, 0, token.length);

                // Send a token to the server if one was generated by initSecContext
                if (token != null) {
                    System.out.println("Will send token of size " + token.length + " from initSecContext.");
                    outStream.writeInt(token.length);
                    outStream.write(token);
                    outStream.flush();
                }

                // If the client is done with context establishment then there will be no more tokens to read in this loop
                if (!context.isEstablished()) {
                    token = new byte[inStream.readInt()];
                    System.out.println(
                            "Will read input token of size " + token.length + " for processing by initSecContext");
                    inStream.readFully(token);
                }
            }

            System.out.println("Context Established! ");
            System.out.println("Client is " + context.getSrcName());
            System.out.println("Server is " + context.getTargName());

        }

        public void sendMessage() throws Exception {
            // Obtain the command-line arguments and parse the port number

            String msg = "Hello Server ";
            byte[] messageBytes = msg.getBytes();
            outStream.writeInt(messageBytes.length);
            outStream.write(messageBytes);
            outStream.flush();

            byte[] token = new byte[inStream.readInt()];
            System.out.println("Will read token of size " + token.length);
            inStream.readFully(token);

            String s = new String(token);
            System.out.println(s);

            System.out.println("Exiting... ");
        }

        public static void main(String[] args) throws Exception {
            TestClient client = new TestClient("root/localhost@HADOOP.COM", "localhost", 9111);
            client.sendMessage();
        }
    }

    public static class TestServer {
        private int localPort;
        private ServerSocket ss;
        private Socket socket = null;

        public TestServer(int port) {
            this.localPort = port;
        }

        public void receive() throws IOException, GSSException {
            this.ss = new ServerSocket(localPort);
            socket = ss.accept();
            DataInputStream in = new DataInputStream(socket.getInputStream());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());
            this.initKerberos(in, out);

            int length = in.readInt();
            byte[] token = new byte[length];
            System.out.println("Will read token of size " + token.length);
            in.readFully(token);
            String s = new String(token);
            System.out.println("Receive Client token: " + s);

            byte[] token1 = "Receive Client Message".getBytes();
            out.writeInt(token1.length);
            out.write(token1);
            out.flush();
        }

        private void initKerberos(DataInputStream in, DataOutputStream out) throws GSSException, IOException {
            GSSManager manager = GSSManager.getInstance();
            GSSContext context = manager.createContext((GSSCredential) null);
            byte[] token;

            while (!context.isEstablished()) {
                token = new byte[in.readInt()];
                System.out.println("Will read input token of size " + token.length + " for processing by acceptSecContext");
                in.readFully(token);

                token = context.acceptSecContext(token, 0, token.length);

                // Send a token to the peer if one was generated by acceptSecContext
                if (token != null) {
                    System.out.println("Will send token of size " + token.length + " from acceptSecContext.");
                    out.writeInt(token.length);
                    out.write(token);
                    out.flush();
                }
            }

            System.out.println("Context Established! ");
            System.out.println("Client is " + context.getSrcName());
            System.out.println("Server is " + context.getTargName());
        }

        public static void main(String[] args) throws IOException, GSSException {
            System.setProperty("java.security.krb5.conf", "experiment/src/main/krb5.conf");
            System.setProperty("java.security.auth.login.config", "experiment/src/main/server.conf");
            System.setProperty("javax.security.auth.useSubjectCredsOnly", "false");
            System.setProperty("sun.security.krb5.debug", "true");

            TestServer server = new TestServer(9111);
            server.receive();
        }
    }
}
```

先运行`Server`程序，再运行`Client`程序，我们将能从输出内容中看到整个通信的过程。

当前web应用成为主流的时候，`Kerberos`如何在`HTTP/HTTPS`协议场景下使用呢？我们又要如何配置，才能运行一套支持认证的Hadoop集群呢？请关注后续文章。

参考：
- [Java官方Demo](https://docs.oracle.com/en/java/javase/13/security/source-code-advanced-security-programming-java-se-authentication-secure-communication-and-single-sig.html)
- [Wiki](https://zh.wikipedia.org/wiki/Kerberos)
- [krb5配置](https://web.mit.edu/kerberos/krb5-1.12/doc/admin/conf_files/krb5_conf.html)
- [kdc配置](https://web.mit.edu/kerberos/krb5-1.12/doc/admin/conf_files/kdc_conf.html)






