---
title: Hadoop安全认证机制 （三）
categories:
- 大数据
- 安全
tags:
- 大数据
- 安全
- hadoop
- kerberos
date: 2019-11-11 20:00:00
---

系列文章：
- [Hadoop安全认证机制 (一)](http://brightliao.me/2019/10/27/hadoop-auth/)
- [Hadoop安全认证机制 (二)](http://brightliao.me/2019/10/30/hadoop-auth-2/)

前面的文章中我们分析了Hadoop安全机制中用到的协议及相关源代码实现，这一篇文章我们主要来看看如何搭建一套安全的Hadoop集群。

简单起见，我们这里的集群所有的组件将运行在同一台机器上。对于keytab的配置，我们也从简，只配置一个kerberos的service账号供所有服务使用。

## 建立测试用例

TDD是敏捷最重要的实践之一，可以有效的帮助我们确定目标，验证目标，它可以带领我们走得又快又稳。跟随TDD的思想，我们先从测试的角度来看这个问题。有了前面的基础知识，假设我们已经有了一套安全的Hadoop集群，那么我们应当可以从集群读写文件，运行MapReduce任务。我们可以编写读写文件的测试用例如下：

<!-- more -->

```java
public class HdfsTest {
    TestConfig testConfig = new TestConfig();

    @Test
    public void should_read_write_files_from_hdfs() throws IOException {
        testConfig.configKerberos();

        Configuration conf = new Configuration();
        conf.addResource(new Path(testConfig.hdfsSiteFilePath()));
        conf.addResource(new Path(testConfig.coreSiteFilePath()));
        UserGroupInformation.setConfiguration(conf);
        UserGroupInformation.loginUserFromKeytab(testConfig.keytabUser(), testConfig.keytabFilePath());

        FileSystem fileSystem = FileSystem.get(conf);
        Path path = new Path("/user/root/input/core-site.xml");
        if (fileSystem.exists(path)) {
            boolean deleteSuccess = fileSystem.delete(path, false);
            assertTrue(deleteSuccess);
        }

        String fileContent = FileUtils.readFileToString(new File(testConfig.coreSiteFilePath()));
        try (FSDataOutputStream fileOut = fileSystem.create(path)) {
            fileOut.write(fileContent.getBytes("utf-8"));
        }

        assertTrue(fileSystem.exists(path));

        try (FSDataInputStream in = fileSystem.open(path)) {
            String fileContentRead = IOUtils.toString(in);
            assertEquals(fileContent, fileContentRead);
        }

        fileSystem.close();
    }
}
```

（完整代码请参考[这里](https://github.com/gmlove/bigdata_conf/blob/master/test/src/test/java/test/HdfsTest.java)）

到这里我们的任务目标就明确了，只要上面的测试能通过，我们的集群就应该搭建好了。

（如果有条件，下面的内容请大家结合代码及参考文档，一边读文章，一边动手实践，否则可能会遗漏很多细节。）

## 建立基本集群

我们先跟随[官网的教程](https://hadoop.apache.org/docs/r2.7.7/hadoop-project-dist/hadoop-common/SingleCluster.html)搭建一个非安全的集群。

这里我选择的Hadoop版本为2.7.7（我这里是为了和实际项目中用到的版本保持一致，大家可以自行尝试其他版本，思路和大部分的脚本都应该是相同的）。我们选择伪分布式模式（Pseudo-Distributed）来进行尝试，这种模式下，每个组件会运行为一个独立的java进程，与真实的分布式环境类似。

我们还是使用容器来进行试验，启动一个容器，并依次运行下面的命令：

```bash
docker run -it --name shd -h shd centos:7 bash
```

在容器中运行下面的命令：

```bash
# 建立并切换到我们的工作目录
mkdir /hd && cd /hd
# 下载软件、解压、进入根目录
yum install wget vim less -y
wget https://archive.apache.org/dist/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz
tar xf hadoop-2.7.7.tar.gz
ln -sv hadoop-2.7.7/ hadoop
cd hadoop
# 配置hadoop
echo shd > etc/hadoop/slaves
cat > etc/hadoop/core-site.xml << EOF
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://0.0.0.0:9000</value>
    </property>
</configuration>
EOF
cat > etc/hadoop/hdfs-site.xml << EOF
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>/hd/data/hdfs/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/hd/data/hdfs/datanode</value>
    </property>
</configuration>
EOF
# 配置ssh，测试：是否能通过`ssh localhost`免密登录
yum install openssh-clients openssh-server -y
echo 'root:screencast' | chpasswd
sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
echo "export VISIBLE=now" >> /etc/profile
ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -P '' && ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -P ''
/usr/sbin/sshd
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 0600 ~/.ssh/authorized_keys
# 安装jdk,并配置环境变量
yum install -y java-1.8.0-openjdk-devel
echo 'export JAVA_HOME=/usr/lib/jvm/java' >> ~/.bashrc
export JAVA_HOME=/usr/lib/jvm/java
# 启动hdfs
bin/hdfs namenode -format
sbin/start-dfs.sh
# 测试
bin/hdfs dfs -mkdir /user
bin/hdfs dfs -mkdir /user/root
bin/hdfs dfs -put etc/hadoop input
bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.7.jar grep input output 'dfs[a-z.]+'
bin/hdfs dfs -cat output/*  # 这里的结果将显示配置文件里面关于dfs的内容
```

到这里我们的非安全的单机模式集群应该就能运行起来了。但是在这个集群里面我们还没法运行分布式任务，因为目前仅仅是一个HDFS分布式文件系统。如果用`jps`查看一下有哪些java进程，将发现我们启动了三个进程`NameNode` `SecondaryNameNode` `DataNode`。

下一步，我们还需要配置并启动用于管理分布式集群任务的关键组件`Yarn`。运行如下这些命令，即可启动`Yarn`：

```bash
# 配置Yarn
cat > etc/hadoop/mapred-site.xml << EOF
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
    <property>
        <name>mapreduce.jobhistory.address</name>
        <value>0.0.0.0:10020</value>
    </property>
    <property>
        <name>mapreduce.jobhistory.webapp.address</name>
        <value>0.0.0.0:19888</value>
    </property>
</configuration>
EOF
cat > etc/hadoop/yarn-site.xml << EOF
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.log-aggregation-enable</name>
        <value>true</value>
    </property>
    <!-- fix node unhealthy issue -->
    <!-- `yarn node -list -all` report node unhealthy with message indicate no disk space (disk space check failed) -->
    <property>
        <name>yarn.nodemanager.disk-health-checker.max-disk-utilization-per-disk-percentage</name>
        <value>99.9</value>
    </property>
    <!-- to fix issue: 'Failed while trying to construct...' (http://blog.51yip.com/hadoop/2066.html) -->
    <property>
         <name>yarn.log.server.url</name>
         <value>http://shd:19888/jobhistory/logs</value>
    </property>
</configuration>
EOF
# 启动Yarn：启动之后我们将能通过`./bin/yarn node -list -all`查看到一个RUNNIN的node
sbin/start-yarn.sh
# 启动History server用于查看应用日志
sbin/mr-jobhistory-daemon.sh start historyserver
# 测试：我们将能看到下面的命令从0%到100%按进度完成。
# 验证：运行`./bin/hadoop dfs -cat output/wc/part-r-00000`还将看到计算出来的结果。
# 验证：运行`./bin/yarn application -list -appStates FINISHED`可以看到已运行完成的任务，及其日志的地址。
bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.7.jar  wordcount input/* output/wc/
```

执行上面的命令启动`Yarn`及`historyserver`之后，我们将发现有三个额外的进程`ResourceManager` `NodeManager` `JobHistoryServer`随之启动了。

如果我们的容器所在主机有一个浏览器可以用，那么我们可以通过访问`http://${SHD_DOCKER_IP}:8088/cluster/apps`将能看到上面的`wordcount`程序运行的状态及日志。这里的`SHD_DOCKER_IP`可以通过下面的命令查找出来。

```
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' shd
```

如果容器是在一个远端的主机上面启动的，我们可以用`ssh tunnel`的方式建立一个代理，通过代理来访问我们的集群。运行命令`ssh -f -N -D 127.0.0.1:3128 ${USER}@${REMOTE_DOCKER_HOST_IP}`即可建立这样的代理。然后我们运行`echo "${SHD_DOCKER_IP} shd" >> /etc/hosts`将容器的主机名加入到我们本地的`hosts`。再使用`firefox`浏览器来配置代理（如下图），这样我们就可以通过本地的`firefox`来访问到远端的集群了。

![Firefox Proxy](/attaches/2019/2019-11-11-hadoop-auth-3/firefox-ssh-tunnel-proxy.png)

我们将能看到如下的web应用，通过这个web应用，我们实际上还可以查询到更多的集群相关的信息。

![App Log](/attaches/2019/2019-11-11-hadoop-auth-3/app-log.png)

可以看到，经过多年的优化，即便是一个非常复杂的分布式系统，我们现在也可以快速的上手了。几乎所有的配置都有相对合理的默认值，我们仅仅需要调整很少的配置。

Hadoop本身内置了很多实用的工具，当我们遇到问题的时候，这些工具可以有效的辅助诊断问题。如果大家经过上面的步骤还是没法通过测试（命令行中的测试）。大家可能可以从以下几个方面去查找问题：

1. 检查各个组件进程是否都启动起来了
2. 检查各个组件的日志，比如，如果`datanode`启动失败，可能我们要查看`logs/hadoop-root-datanode-shd.log`日志做进一步分析
3. 使用`bin/yarn node -list -all`检查`node`的状态
4. 检查最终生成的配置`http://172.17.0.12:8042/conf`是否是我们所希望的，比如我们可能由于拼写错误导致配置不对


## Kerberos安全配置

在本系列[第一篇文章](http://brightliao.me/2019/10/27/hadoop-auth/)中，我们尝试了搭建一个kerberos认证服务器，这里我们可以用与之前一致的方式先搭建起一个kerberos认证服务器。需要的执行脚本如下：

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
```

## 配置Hadoop安全支持

前面我们分析了Kerberos的运行原理，及Hadoop的相关源代码，可以知道，为了启动安全支持，每一个集群节点的每一个hadoop组件都将需要单独的Kerberos账号及其keytab文件，每个组件最好还能用不同的账户启动。这里由于我们使用伪分布式模式来部署集群，所有的组件都运行在同一个节点，简单起见，我们这里将使用root账号来启动集群，并让所有的组件使用同一个kerberos账号。

首先我们生成账号如下：

```bash
mkdir /hd/conf/
# 生成hadoop集群需要的账号
kadmin.local addprinc -randkey root/shd@HADOOP.COM
kadmin.local addprinc -randkey HTTP/shd@HADOOP.COM
kadmin.local xst -k /hd/conf/hadoop.keytab root/shd@HADOOP.COM HTTP/shd@HADOOP.COM
# 生成测试用的普通账号
kadmin.local addprinc -randkey root@HADOOP.COM
kadmin.local xst -k /hd/conf/root.keytab root@HADOOP.COM
```

接下来我们来完成hadoop的配置，由于配置文件内容比较多，我统一整理到了github的一个repo中，下面的配置将主要通过copy这些文件来生成，而辅以说明主要修改的地方。如果大家有兴趣知道确切的修改之处，可以备份这些文件，然后用diff来查看修改，或者用git对配置文件进行版本管理，然后查看修改。

### 配置集群

#### 配置`core-site.xml`

```bash
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hadoop/etc/hadoop/core-site.xml -O etc/hadoop/core-site.xml
sed -i 's/hd01-7/shd/g' etc/hadoop/core-site.xml
```

这里主要加入的配置项及其解释如下：

```
hadoop.proxyuser.root.hosts=*           # 配置root用户（组件启动时认证的kerberos账户）可以以任意客户端认证过的用户（proxy user）来执行操作，详见：https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/Superusers.html
hadoop.proxyuser.root.groups=*
hadoop.proxyuser.HTTP.hosts=*
hadoop.proxyuser.HTTP.groups=*
hadoop.security.authorization=true
hadoop.security.authentication=kerberos
```

#### 配置`hdfs-site.xml`

```bash
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hadoop/etc/hadoop/hdfs-site.xml -O etc/hadoop/hdfs-site.xml
sed -i 's/hd01-7/shd/g' etc/hadoop/hdfs-site.xml
```

这里主要加入的配置项如下：

```
dfs.block.access.token.enable=true
dfs.namenode.keytab.file=/hd/conf/hadoop.keytab
dfs.namenode.kerberos.principal=root/_HOST@HADOOP.COM
dfs.namenode.kerberos.internal.spnego.principal=HTTP/_HOST@HADOOP.COM
dfs.web.authentication.kerberos.principal=HTTP/_HOST@HADOOP.COM
dfs.web.authentication.kerberos.keytab=/hd/conf/hadoop.keytab
dfs.datanode.keytab.file=/hd/conf/hadoop.keytab
dfs.datanode.kerberos.principal=root/_HOST@HADOOP.COM
dfs.datanode.address=0.0.0.0:1004
dfs.datanode.http.address=0.0.0.0:1006
dfs.journalnode.keytab.file=/hd/conf/hadoop.keytab
dfs.journalnode.kerberos.principal=root/_HOST@HADOOP.COM
dfs.journalnode.kerberos.internal.spnego.principal=HTTP/_HOST@HADOOP.COM
```

#### 配置`mapred-site.xml`

```bash
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hadoop/etc/hadoop/mapred-site.xml -O etc/hadoop/mapred-site.xml
sed -i 's/hd01-7/shd/g' etc/hadoop/mapred-site.xml
```

这里主要加入的配置项如下：

```
mapreduce.jobhistory.address=shd:10020
mapreduce.jobhistory.webapp.address=shd:19888
mapreduce.jobhistory.principal=root/_HOST@HADOOP.COM
mapreduce.jobhistory.keytab=/hd/conf/hadoop.keytab
```

#### 配置`yarn-site.xml`

```bash
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hadoop/etc/hadoop/yarn-site.xml -O etc/hadoop/yarn-site.xml
sed -i 's/hd01-7/shd/g' etc/hadoop/yarn-site.xml
```

这里主要加入的配置项如下：

```
yarn.resourcemanager.principal=root/_HOST@HADOOP.COM
yarn.resourcemanager.keytab=/hd/conf/hadoop.keytab
yarn.resourcemanager.webapp.https.address=${yarn.resourcemanager.hostname}:8090
yarn.nodemanager.principal=root/_HOST@HADOOP.COM
yarn.nodemanager.keytab=/hd/conf/hadoop.keytab
yarn.web-proxy.principal=root/_HOST@HADOOP.COM
yarn.web-proxy.keytab=/hd/conf/hadoop.keytab
```


#### 配置`hadoop-env.sh`

```bash
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hadoop/etc/hadoop/hadoop-env.sh -O etc/hadoop/hadoop-env.sh
```

主要加入的配置项如下：

```bash
export JSVC_HOME=/usr/bin             # 指定jsvc的路径，以便运行安全模式的datanode
export HADOOP_JAAS_DEBUG=true         # 开启Kerberos认证的debug日志
export HADOOP_OPTS="-Djava.net.preferIPv4Stack=true -Dsun.security.krb5.debug=true -Dsun.security.spnego.debug"  # 开启Kerberos认证的debug日志
export HADOOP_SECURE_DN_USER=root     # 运行安全模式的datanode组件的用户
export HADOOP_HDFS_USER=root          # 运行hdfs组件的用户
```

#### 修复启动脚本

由于我们开启了`Kerberos`的调试日志，原来的脚本需要稍加修改才能使用。执行脚本如下：

```bash
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hadoop/sbin/stop-dfs.sh -O sbin/stop-dfs.sh
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hadoop/sbin/start-dfs.sh -O sbin/start-dfs.sh
```

主要修改为将通过`hdfs getconf SOME_CONFIG`命令拿到的配置，修改为通过`hdfs getconf SOME_CONFIG >/dev/null | tail -n 1`去获取配置。这里的`tail -n 1`可以去掉命令运行中的`Kerberos`调试日志。

### 启动集群

启动集群并运行测试如下：

```bash
yum install -y apache-commons-daemon-jsvc.x86_64     # 安装jsvc以便可以用安全模式启动datanode，详见：https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SecureMode.html#Secure_DataNode
sbin/start-dfs.sh && ./sbin/start-secure-dns.sh && sbin/start-yarn.sh && sbin/mr-jobhistory-daemon.sh start historyserver     # 依次启动集群的其他组件
# 测试：我们将能看到下面的命令从0%到100%按进度完成。
# 验证：运行`./bin/hadoop dfs -cat output/wc/part-r-00000`还将看到计算出来的结果。
# 验证：运行`./bin/yarn application -list -appStates FINISHED`可以看到已运行完成的任务，及其日志的地址。
bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.7.jar  wordcount input/* output/wc/
```

如果我们无需再测试了，可以用以下命令停止集群：

```
sbin/stop-dfs.sh && ./sbin/stop-secure-dns.sh && sbin/stop-yarn.sh && sbin/mr-jobhistory-daemon.sh stop historyserver
```


### 运行最初定义的测试

执行命令如下：

```bash
# 加入相关的hosts
SHD_DOCKER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' shd)
echo "${SHD_DOCKER_IP} shd kdc-server kdc-server.hadoop.com" >> /etc/hosts
# 下载源代码
git clone https://github.com/gmlove/bigdata_conf.git
# 更新配置文件
cd bigdata_conf
cd test/src/test && mv resources resources.1
docker cp shd:/hd/hadoop/etc/hadoop/hdfs-site.xml ./resources/
docker cp shd:/hd/hadoop/etc/hadoop/core-site.xml ./resources/
docker cp shd:/hd/hadoop/etc/hadoop/yarn-site.xml ./resources/
docker cp shd:/etc/krb5.conf ./resources/
docker cp shd:/hd/conf/root.keytab ./resources/
cp ./resources.1/log4j.properties ./resources/
# 运行测试
mvn -Dtest=test.HdfsTest test
```

运行上面的命令，我们将能看到测试成功执行。

### 如果容器在一个远端的主机上启动

如果容器是在一个远端的主机上面启动的，我们还是可以通过`ssh tunnel`的方式将远端的端口映射到本地来执行此测试。不过，我们需要对前面步骤中的内容作出一些修改。主要的修改是将涉及到的`hostname`配置从`shd`改为`localhost`。这是由于在做端口映射之后，所有的服务均会通过`localhost`来访问，如果我们还是用`shd`，则集群在进行`Kerberos`认证时，主机名验证会出错。

这个任务还是挺有意思的，可以有效的检验我们对于网络、Hadoop集群、`Kerberos`认证机制等的理解。有兴趣的小伙伴可以尝试实验一下，本文就不赘述了。

## 总结

搭建一套安全的hadoop集群，确实不容易，即使我们只是一个伪分布式环境，还做了各种配置简化，也需要花费一番功夫，更别提真正在生产环境中搭建一套集群了。如果是生产可用，我们可能还需要关心机架、集群网络情况、稳定性、性能、跨地域高可用、不宕机升级等等一系列的问题。在实际企业应用中，这些大数据基础设施运维实际上是一个比较复杂的工作，这些工作更可能是由一个单独的运维团队去完成的。这里我们所完成的例子的主要价值不在于生产可用，而在于它可以帮助我们理解hadoop集群的安全机制，以便指导我们日常的开发工作。另一个价值是，这里的例子实际上完全可以作为我们平时测试用的一套小集群，简单而又功能完整，我们完全可以将这里完成的工作制作为一个docker镜像（后续文章将尝试制作此镜像），随时启动这样一套集群，这对于我们测试一些集群集成问题时将带来很大的便利。

大家如果有自己实践，相信在这个过程中可能还会碰到其他的问题，欢迎留言交流，一起学习。

在这篇文章里，我们搭建了一个安全的hadoop集群，那么大数据相关的其他组件应该要如何安全的和hadoop集群进行整合呢？下一篇文章我们将选取几个典型的组件来分析并进行实践，欢迎持续关注。zbd

## 参考

- Hadoop官方文档`Secure Mode`：https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SecureMode.html
- Hadoop官方文档`Proxy User`：https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/Superusers.html
