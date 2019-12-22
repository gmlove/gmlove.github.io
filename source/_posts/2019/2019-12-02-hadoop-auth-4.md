---
title: Hadoop安全认证机制 （四）
categories:
- 大数据
- 安全
tags:
- 大数据
- 安全
- hadoop
- kerberos
date: 2019-12-02 23:00:00
---

系列文章：
- [Hadoop安全认证机制 (一)](http://brightliao.me/2019/10/27/hadoop-auth/)
- [Hadoop安全认证机制 (二)](http://brightliao.me/2019/10/30/hadoop-auth-2/)
- [Hadoop安全认证机制 (三)](http://brightliao.me/2019/11/11/hadoop-auth-3/)

前面的文章中，我们搭建了一套安全的`hadoop`集群，并建立了一个对应的测试。结合相关的基础知识，我们应该对安全的`hadoop`集群有了一定的认识。本文主要关注如何将大数据其他组件安全地与`hadoop`进行集成。我们将关注这几个组件：`hive` `hbase` `spark` `livy`。

## `hive`

首先来看`hive`，`hive`是最初由`facebook`发起的基于`hadoop`大数据架构的数据仓库。`hive`直接将`hdfs`变为了一个支持`sql`的数据存储。时至今日，`hive`已成为企业大数据最基础的组件之一，各种上层的组件均和`hive`有良好的集成。

采用跟之前同样的思路，我们先来建立一个测试，如下：

<!-- more -->

```java
public class HiveTest {
    TestConfig testConfig = new TestConfig();

    @Test
    public void should_connect_to_hive_and_execute_query() throws IOException, SQLException {
        testConfig.configKerberos();
        org.apache.hadoop.conf.Configuration conf = new
                org.apache.hadoop.conf.Configuration();
        conf.set("hadoop.security.authentication", "Kerberos");
        UserGroupInformation.setConfiguration(conf);
        UserGroupInformation.loginUserFromKeytab(testConfig.keytabUser(), testConfig.keytabFilePath());

        String url = testConfig.hiveUrl();
        Connection conn = DriverManager.getConnection(url);
        Statement statement = conn.createStatement();

        statement.execute("create database if not exists t");
        statement.execute("drop table if exists t.t");
        statement.execute("create table t.t (a int)");
        statement.execute("insert into table t.t values (1), (2)");

        ResultSet resultSet = statement.executeQuery("desc t.t");
        resultSet.next();
        assertEquals("a", resultSet.getString("col_name"));
        assertEquals("int", resultSet.getString("data_type"));
        assertEquals("", resultSet.getString("comment"));

        resultSet = statement.executeQuery("select * from t.t");
        resultSet.next();
        assertEquals(1, resultSet.getInt("a"));
        resultSet.next();
        assertEquals(2, resultSet.getInt("a"));

        conn.close();
    }
}
```

（完整代码请参考[这里](https://github.com/gmlove/bigdata_conf/blob/master/test/src/test/java/test/HiveTest.java)）

`hive`的运行时包括两个核心组件，存储元数据的`metastore`及对外提供`sql`操作服务`hiveserver`。在这里我们用使用最广泛`mysql`作为`metastore`的存储。

执行脚本如下，即可搭建并运行`metastore`及`hiveserver`:

```bash
# 进入我们之前创建好的容器
docker exec -it shd bash

# 下载、解压、配置
cd /hd
wget https://mirrors.tuna.tsinghua.edu.cn/apache/hive/hive-1.2.2/apache-hive-1.2.2-bin.tar.gz
tar xf apache-hive-1.2.2-bin.tar.gz
ln -sv apache-hive-1.2.2-bin hive

# 安装metastore
yum install -y mariadb-server
/usr/libexec/mariadb-prepare-db-dir mariadb.service
/usr/bin/mysqld_safe --basedir=/usr 2>&1 1>hive/hive.metastore.mysql.log &
mysql -uroot -e "create user 'hive'@'localhost' identified by '123456'; grant all on *.* to hive@'localhost';"
# 测试metastore连接
mysql -uhive -p123456 -e 'select now()'

# 配置、启动hive
wget https://repo1.maven.org/maven2/mysql/mysql-connector-java/5.1.48/mysql-connector-java-5.1.48.jar -O hive/lib/mysql-connector-java-5.1.48.jar
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hive/conf/hive-site.xml -O hive/conf/hive-site.xml
cd hive && bin/hive --service metastore 2>&1 1>hive.metastore.service.log &

# 用hive的命令行工具测试 (下面的命令需要成功运行)
cd hive && bin/beeline
> !connect jdbc:hive2://localhost:10000/default;principal=root/shd@HADOOP.COM
Enter username for jdbc:hive2://localhost:10000/default;principal=root/shd@HADOOP.COM: (这里直接Enter)
Enter password for jdbc:hive2://localhost:10000/default;principal=root/shd@HADOOP.COM: (这里直接Enter)
> create database if not exists t;
> drop table if exists t.t;
> create table t.t (a int);
> insert into table t.t values (1), (2);
> select * from t.t;
```

我们这里修改的配置如下：

```properties
hive.server2.authentication=KERBEROS
hive.server2.authentication.kerberos.principal=root/_HOST@HADOOP.COM
hive.server2.authentication.kerberos.keytab=/hd/conf/hadoop.keytab

hive.metastore.sasl.enabled=true
hive.metastore.kerberos.keytab.file=/hd/conf/hadoop.keytab
hive.metastore.kerberos.principal=root/_HOST@HADOOP.COM

javax.jdo.option.ConnectionURL=jdbc:mysql://localhost/hive_remote?createDatabaseIfNotExist=true
javax.jdo.option.ConnectionDriverName=com.mysql.jdbc.Driver
javax.jdo.option.ConnectionUserName=hive
javax.jdo.option.ConnectionPassword=123456
```

运行最初我们定义的测试如下：（接上篇中对应的测试运行部分）

```bash
cd bigdata_conf/test
cat > src/test/resources/hive-site.xml <<EOF
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>hive.server2.authentication</name>
    <value>KERBEROS</value>
  </property>
  <property>
    <name>hive.server2.authentication.kerberos.principal</name>
    <value>root/shd@HADOOP.COM</value>
  </property>
  <property>
    <name>hive.metastore.sasl.enabled</name>
    <value>true</value>
  </property>
  <property>
    <name>hive.metastore.kerberos.principal</name>
    <value>root/shd@HADOOP.COM</value>
  </property>
</configuration>
EOF
mvn -DshdHost=shd -Dtest=test.HiveTest test
```

到这里一个安全的`hive`搭建和测试就完成了。

## `hbase`

`hbase`是一个分布式大规模数据存储组件，支持随机的、实时的读写超过数十亿行的超大数据库。`hbase`基于google的论文《Bigtable: A Distributed Storage System for Structured Data》设计实现。我们来看看如何搭建一个安全的`hbase`数据库。

对于一个安全`hbase`的读写，我们可以建立测试如下：

```java
public class HbaseTest {
    TestConfig testConfig = new TestConfig();

    @Test
    public void should_read_write_hbase() throws IOException {
        testConfig.configKerberos();
        Configuration config = HBaseConfiguration.create();
        config.addResource(new Path(testConfig.hbaseSiteFilePath()));
        UserGroupInformation.setConfiguration(config);
        UserGroupInformation.loginUserFromKeytab(testConfig.keytabUser(), testConfig.keytabFilePath());

        TableName tableName = TableName.valueOf("test");

        Connection connection = ConnectionFactory.createConnection(config);
        Admin admin = connection.getAdmin();

        if (admin.tableExists(tableName)) {
            admin.deleteTable(tableName);
        }

        String family1 = "Family1";
        String family2 = "Family2";
        HTableDescriptor tableDescriptor = new HTableDescriptor(tableName);
        tableDescriptor.addFamily(new HColumnDescriptor(family1));
        tableDescriptor.addFamily(new HColumnDescriptor(family2));
        admin.createTable(tableDescriptor);

        Put p = new Put(Bytes.toBytes("row1"));
        String qualifier1 = "Qualifier1";
        p.addColumn(family1.getBytes(), qualifier1.getBytes(), "value1".getBytes());
        p.addColumn(family2.getBytes(), qualifier1.getBytes(), "value2".getBytes());

        Table table = connection.getTable(tableName);
        table.put(p);

        Get g = new Get(Bytes.toBytes("row1"));

        assertEquals("value1", new String(table.get(g).getValue(family1.getBytes(), qualifier1.getBytes())));
        connection.close();
    }
}
```

（完整代码请参考[这里](https://github.com/gmlove/bigdata_conf/blob/master/test/src/test/java/test/HbaseTest.java)）

`hbase`的运行时包括三个核心组件，存储元数据的`Master`、存储数据的`RegionServers`及一个分布式协调器`ZooKeeper`。在这里，我们将部署`ZooKeeper`作为一个独立的大数据组件运行，以便将来当我们要引入其他的基于`ZooKeeper`的组件时可以直接使用。 

运行下面的命令，即可搭建并运行一个安全的`ZooKeeper`:

```bash
wget https://archive.apache.org/dist/zookeeper/zookeeper-3.5.5/apache-zookeeper-3.5.5-bin.tar.gz
tar xf /
ln -sv apache-zookeeper-3.5.5-bin zookeeper
# 创建对应的key
kadmin.local addprinc -randkey zookeeper/shd@HADOOP.COM
kadmin.local xst -k /hd/conf/zookeeper.keytab zookeeper/shd@HADOOP.COM
# 配置、启动
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/zookeeper/bin/zkEnv.sh -O zookeeper/bin/zkEnv.sh
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/zookeeper/conf/jaas.conf -O zookeeper/conf/jaas.conf
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/zookeeper/conf/client-jaas.conf -O zookeeper/conf/client-jaas.conf
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/zookeeper/conf/zoo.cfg -O zookeeper/conf/zoo.cfg
sed -i "s/__HOST__/shd/g" conf/jaas.conf
sed -i "s/__HOST__/shd/g" conf/client-jaas.conf
cd zookeeper && bin/zkServer.sh start
# 测试如下：
cd zookeeper && bin/zkCli.sh
> create /test test
> get /test
```

接下来是`hbase`，运行命令如下：

```bash
# 下载、配置hbase
wget http://mirrors.tuna.tsinghua.edu.cn/apache/hbase/hbase-1.3.6/hbase-1.3.6-bin.tar.gz
tar xf hbase-1.3.6-bin.tar.gz
ln -sv hbase-1.3.6 hbase
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hbase/conf/hbase-site.xml -O hbase/conf/hbase-site.xml
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hbase/conf/hbase-env.sh -O hbase/conf/hbase-env.sh
wget https://raw.githubusercontent.com/gmlove/bigdata_conf/master/auth/hbase/conf/jaas.conf -O hbase/conf/jaas.conf
echo 'shd' > hbase/conf/regionservers
sed -i "s/__HOST__/shd/g" conf/hbase-site.xml
sed -i "s/__HOST__/shd/g" conf/jaas.conf
# 启动
cd hbase
./bin/start-hbase.sh

# 测试
./bin/hbase shell
> create 'member', 'id','address','info'
> put 'member', 'debugo', 'id', '1'
> get 'member', 'debugo'
```

运行最初我们定义的测试如下：（接上篇中对应的测试运行部分）

```bash
cd bigdata_conf/test
docker cp shd:/hd/hbase/conf/hbase-site.xml ./resources/
mvn -DshdHost=shd -Dtest=test.HbaseTest test
```

到这里一个安全的`hase`搭建和测试就完成了。

## `Spark`

`Spark`是一个独立的通用的高性能分布式计算引擎。相比基于`MapReduce`计算模型的`hive`，`Spark`设计了一套高效的`DAG`来优化计算流，有效的防止了多余的中间数据存储。由于`Spark`提供了更高的计算效率，它逐渐成了当前最流行的计算框架。

`Spark`其实是作为一个工具库来与`Hadoop`大数据集群进行集成的。我们直接依赖`Spark`的库就可以使用，无需任何配置。

我们可以建立测试如下：

```java
public class SparkTest {

    TestConfig testConfig = new TestConfig();

    @Test
    public void should_be_able_to_read_hive_from_spark() throws IOException {
        testConfig.configKerberos();
        org.apache.hadoop.conf.Configuration conf = new
                org.apache.hadoop.conf.Configuration();
        conf.set("hadoop.security.authentication", "Kerberos");
        UserGroupInformation.setConfiguration(conf);
        UserGroupInformation.loginUserFromKeytab(testConfig.keytabUser(), testConfig.keytabFilePath());
        SparkSession spark = SparkSession
                .builder()
                .appName("Simple Spark Example")
                .master("yarn")
                .enableHiveSupport()
                .config("spark.sql.warehouse.dir", testConfig.sparkSqlWarehouseDir())
                .config("hive.metastore.uris", testConfig.hiveMetastoreUrl())
                .getOrCreate();

        spark.sql("create database if not exists t");
        spark.sql("drop table if exists t.t");
        spark.sql("create table t.t (a int)");
        spark.sql("insert into table t.t values (1), (2)");
        spark.sql("desc t.t").show();
        spark.sql("select * from t.t").show();

        spark.stop();
        spark.close();
    }
}
```

（完整代码请参考[这里](https://github.com/gmlove/bigdata_conf/blob/master/test/src/test/java/test/SparkTest.java)）

运行以上测试：

```bash
cd bigdata_conf/test
mvn -DshdHost=shd -Dtest=test.SparkTest test
```

如果我们想使用`spark-shell`来交互式的探索`spark`，则我们需要下载spark并做一定的配置。

执行脚本如下：

```bash
wget https://archive.apache.org/dist/spark/spark-2.1.0/spark-2.1.0-bin-hadoop2.7.tgz
tar xf spark-2.1.0-bin-hadoop2.7.tgz
ln -sv spark-2.1.0-bin-hadoop2.7 spark
export HADOOP_HOME=/hd/hadoop
export HIVE_HOME=/hd/hive
export HADOOP_CONF_DIR=/hd/hadoop/etc/hadoop
cp hive/conf/hive-site.xml spark/conf/
wget https://repo1.maven.org/maven2/mysql/mysql-connector-java/5.1.48/mysql-connector-java-5.1.48.jar -O spark/jars/mysql-connector-java-5.1.48.jar

# 测试：（下面的spark程序将能计算出PI的近似值）
cd spark
./bin/spark-submit \
    --class org.apache.spark.examples.SparkPi \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 512M \
    --num-executors 1 \
    examples/jars/spark-examples_2.11-2.1.0.jar 1000

# 运行spark-shell进入交互式环境
./bin/spark-shell --master yarn
> new org.apache.spark.sql.SQLContext(sc).sql("show databases").collectAsList()
```

## `Livy`

`Livy`将运行`spark`应用的这一能力封装为了一个服务，这样我们就可以更方便的进行统一资源管理。

对于一个安全的`Livy`服务，我们可以编写对应的测试如下：

```java
public class LivyTest {
    private static Logger log = LoggerFactory.getLogger(LivyTest.class);

    TestConfig testConfig = new TestConfig();

    @Test
    public void should_submit_and_run_job_through_livy() throws IOException, URISyntaxException, InterruptedException, ExecutionException {
        testConfig.configKerberos();
        LivyClient client = new LivyClientBuilder()
                .setURI(new URI(testConfig.livyUrl()))
                .setConf("livy.client.http.spnego.enable", "true")
                .setConf("livy.client.http.auth.login.config", testConfig.jaasConfPath())
                .setConf("livy.client.http.krb5.conf", testConfig.krb5FilePath())
                .setConf("livy.client.http.krb5.debug", "true")
                .build();
        try {
            String piJar = testConfig.sparkPiJarFilePath();
            log.info("Uploading {} to the Spark context...", piJar);
            client.uploadJar(new File(piJar)).get();

            int samples = 10000;
            log.info("Running PiJob with {} samples...\n", samples);
            double pi = client.submit(new PiJob(samples)).get();

            log.info("Pi is roughly: " + pi);
            assertEquals(3, Double.valueOf(pi).intValue());
        } finally {
            client.stop(true);
        }
    }
}
```

（完整代码请参考[这里](https://github.com/gmlove/bigdata_conf/blob/master/test/src/test/java/test/LivyTest.java)及[这里](https://github.com/gmlove/bigdata_conf/blob/master/test/src/main/java/test/PiJob.java)）

我们曾在[Hadoop安全认证机制 (二)](http://brightliao.me/2019/10/30/hadoop-auth-2/)中介绍过`Livy`的安全机制。结合当时的原理介绍，我们只需要运行下面的脚本即可以配置好一个安全的`Livy`:

```bash
yum install -y unzip
ln -sv livy-0.5.0-incubating-bin livy
cat >> livy/conf/livy.conf << EOF
livy.impersonation.enabled = true
livy.server.auth.type = kerberos
livy.server.auth.kerberos.keytab = /hd/conf/hadoop.keytab
livy.server.auth.kerberos.principal = HTTP/shd@HADOOP.COM
livy.server.launch.kerberos.keytab = /hd/conf/hadoop.keytab
livy.server.launch.kerberos.principal = root/shd@HADOOP.COM
EOF

# 启动livy
cd livy && ./bin/livy-server start
```

运行上面的`java`测试用例：

```bash
cd bigdata_conf/test
cat >>src/test/resources/jaas.conf <<EOF
com.sun.security.jgss.krb5.initiate {
  com.sun.security.auth.module.Krb5LoginModule required debug=true
  refreshKrb5Config=true
  doNotPrompt=true
  useKeyTab=true
  keyTab="src/test/resources/root.keytab"
  principal="root";
};
EOF
cp -v /root/dev/projects/tmp/bigdata_conf/test/src/test/resources{.1,}/spark-pi.jar
mvn -DshdHost=shd -Dtest=test.LivyTest test
```

## 制作`docker`镜像

到这里我们常用的几个基础组件的安全配置就介绍完了。为便于重现和复用这样一套环境，我们可以制作一个`docker`镜像。我这里已经将此镜像制作完成，并上传到了`docker hub`[这里](https://hub.docker.com/repository/docker/brightlgm/bigdata-auth/general)，大家可以直接下载使用，或者也参考这里的`Dockerfile`自行制作镜像。

有时候由于我们本地的资源不够运行集群的所有组件，我们可以在另一台机器上面通过容器运行此集群，然后通过`ssh tunnel`的方式将需要的端口映射到本地。这样本地测试时就可以直接连接`localhost`进行测试了。我也将此镜像制作完成，并上传到了`docker hub`[这里](https://hub.docker.com/repository/docker/brightlgm/bigdata-auth-localhost/general)。

对于`localhost`模式，我们需要映射的端口如下：

``` bash
# kdc
ssh -L localhost:1802:localhost:1802 root@rshd

## test hbase:
# hbase zookeeper
ssh -L localhost:2181:localhost:2181 root@rshd
# hbase master
ssh -L localhost:16000:localhost:16000 root@rshd
# hbase region server
ssh -L localhost:16201:localhost:16201 root@rshd

## test hive:
# hive
ssh -L localhost:10000:localhost:10000 root@rshd

## test hdfs:
# hdfs namenode
ssh -L localhost:9000:localhost:9000 root@rshd
# hdfs datanode
sudo ssh -i /root/.ssh/id_rsa -L localhost:1004:localhost:1004 root@rshd -p 12822

## test livy
ssh -L localhost:8998:localhost:8998 root@rshd

## test spark with hive
# hdfs namenode
ssh -L localhost:9000:localhost:9000 root@rshd
# hdfs datanode
sudo ssh -i /Users/gmliao/.ssh/id_rsa -L localhost:1004:localhost:1004 root@rshd -p 12822
# hive metastore
ssh -L localhost:9083:localhost:9083 root@rshd
```

对于前面几个组件的运维，我还将它们组织到了一个`Makefile`中，请参考[这里](https://github.com/gmlove/bigdata_conf/blob/master/auth/Makefile)。有了这个脚本我们可以通过执行`make start-hbase`即可启动`hbase`了。其他的运维工具请参考`Makefile`源代码。

## 总结

本系列文章通过以下四个方面较全面的介绍了`hadoop`的安全机制：

1. Kerberos协议介绍及实践
2. Kerberos协议发展及Hadoop相关源码分析
3. Hadoop安全集群搭建及测试
4. 周边工具的安全支持

通过这些介绍，相信大家对于`hadoop`的安全机制有了一定的了解了。本系列文章也将告一段落。当然我们在实际使用过程中可能还会遇到其他的问题，如有相关问题，欢迎大家留言交流。








