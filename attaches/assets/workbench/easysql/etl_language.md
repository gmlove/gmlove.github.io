
第一，必须坚持社会主义道路
第二，必须坚持无产阶级专政
第三，必须坚持中国共产党的领导
第四，必须坚持马列主义、毛泽东思想

```python


    def latest_valid_orders(order: DataFrame):
        id_window = Window.partitionBy("id").orderBy(desc("update_time"))
        order = order.withColumn('n', row_number().over(id_window))
        order = order.filter((order.n == 1) & (order.canceled is False))
        return order


        def latest_valid_products1(product: DataFrame):
            product.createOrReplaceTempView('product')
            return product.sql_ctx.sql('''
            select * from (
                select *, row_number() over(partition by id order by update_time desc) as n
                from product 
            ) t
            where t.n = 1 and deleted = false
            ''')


        def product_order_count():
            latest_orders = latest_valid_orders(spark.sql('select * from orders'))
            products = latest_valid_products1(spark.sql('select * from products'))
            cond = latest_orders.product_id == products.id
            order_with_product = latest_orders.join(products, cond, 'left')
            product_order_count = order_with_product \
                .groupby(order_with_product.product_id) \
                .count()
            return product_order_count


        def some_func():
            a = 1
            b = B()
            c = b.as_int() + a

```

```java

    public static int addInt(int a, int b) {
        int sum = 0;
        sum = a + b;
        return sum;
    }

```


```go

    func add(x int, y int) int {
        var c int = x + y;
        return c
    }

```


```c

    int add(int x, int y) {
        int sum = x + y;
        return sum;
    } 

```

```python

    def add(a: str, b: str) -> str:
        return a + b

    def func_with_var_args(a: str, b: str, *args):
        return ', '.join([a, b] + args)

    def ensure_partition_exists(table: str, partition_value: str):
        partitions = spark.sql(f'show partitions {table}')
        partition_values = extract_partition_values(partitions)
        if partition_value not in partition_values:
            raise Exception(f'partition {partition_value} not exists!')

    def send_wechat_alert(msg: str):
        alert_service.send_alert_to_team(msg, team_id)



```

```SQL

    -- target=variables
    ...

    -- target=cache.table_a
    select
        *
    from some_db.table_a

    -- target=temp.table_b
    select
        *
    from some_db.table_b

    -- target=temp.table_c
    select
        *
    from some_db.table_c

    -- target=output.some_db.some_table
    select
        *
    from table_a a
        left join table_b b on a.id=b.id
        left join table_c c on a.id=c.id


    -- target=variables, if=bool()
    -- 定义名为a b sep的变量，取值为SQL的执行结果值
    select
        '2021-01-01'                    as a
        , 'some sample alert message'   as b
        , ':'                           as sep

    -- target=func.ensure_partition_exists(some_db.some_table, ${a})
    -- 确保表some_db.some_table有一个值为2021-01-01的分区，如果没有，抛出异常，中断执行

    -- target=func.send_alert(${func_with_var_args(${b}, ${sep}, ${a})})
    -- 向微信发送告警消息，消息内容为函数func_with_var_args的调用结果，即组合后的消息

    -- target=cache.table_a
    -- 创建临时表，引用变量作为条件，调用函数，用其返回值作为条件
    select
        '${add(${a}, ${b})}'            as a
        , *
    from some_db.some_table
    where col_b = ${add(1, 2)}
        and col_d = '${a}'




    -- target=temp.table_b
    -- 定义临时表，通过上面定义的缓存表table_a经过一定的转换得到
    select
        a.*
        , b.*
    from some_db.table_b b
        join table_a a on a.col_a = b.col_b

    -- target=output.some_db.some_table
    select * from table_b


    -- target=temp.some_temp_table, if=partition_exists(some_db.some_table, ${data_date})
    select 
        *
    from some_db.some_table
    where data_date='${data_date}'

    -- target=variables
    select
        count(1)                                        as illegal_data_count
        , 'Found some data with null primary key'       as illegal_data_message
    from some_db.some_table
    where primary_key is null

    -- target=func.send_alert(${illegal_data_message}), if=greater($(illegal_data_count), 0)

    -- target=func.log_something(...), if=not_eq(${env}, prod)


    -- target=template.dim_cols
    -- 定义可复用的维度字段列表为一个模板
    product,
    data_date

    -- target=template.join_conditions
    -- 定义可复用的join条件为一个模板
    ((dim.product is null and #{right_table}.product is null)
        or (dim.product = #{right_table}.product))
    and dim.data_date = #{right_table}.data_date

    -- target=temp.dim
    -- 在SQL代码中引用维度字段模板
    select @{dim_cols}
    from order_count
    union
    select @{dim_cols}
    from sale_amount


    -- include=snippets/dim_etl_snippets.sql

    -- target=temp.result
    -- 在SQL代码中引用join条件模板，并传入模板参数
    select
        dim.product,
        dim.data_date,
        order_count,
        sale_amount
    from dim
        left join order_count
            on @{join_conditions(right_table=order_count)}
        left join sale_amount
            on @{join_conditions(right_table=sale_amount)}

    -- target=variables
    select
        count(1) as order_count
    from order

    -- target=log.order_count
    -- 打印日志，记录当前ETL处理的订单数
    select ${order_count} as order_count

    -- target=log.sample_order
    -- 打印日志，记录当前ETL处理的订单的一条样例数据
    select
        *
    from order
    limit 1

    -- target=temp.order_with_product
    -- 订单和商品表进行连接查询
    select
        *
    from order o
        join product p on o.product_id=p.id

    -- target=variables
    select
        count(1) as order_count_after_join
    from order_with_product

    -- target=check.order_count_must_match_after_joined_with_product
    -- 与连接商品表之后，检查订单数量是否有变化
    select
        ${order_count}                  as expected
        , ${order_count_after_join}     as actual
    from order_with_product

    -- target=check.equal(${order_count}, ${order_count_after_join})
    -- 与连接商品表之后，检查订单数量是否有变化



```


