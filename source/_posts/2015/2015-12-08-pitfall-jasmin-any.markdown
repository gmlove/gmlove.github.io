---
layout: post
title: "jasmine.any之坑"
date: 2015-12-08 23:29:47 +0800
comments: true
categories: 
- 那些年我们踩过的坑
---

坑说：`Jasmine`的`any(Object)`不能替代`any({premitive type})`，可以考虑使用`anything()`

坑位：使用`toHaveBeenCalledWith`测试函数被调用时的参数。

当参数列表太长（如：`func(p1, p2, p3){...}`）的时候，往往只需要验证某一部分参数正确性，这个时候使用`any()`。
<!-- more -->
坑点示例：

```javascript
it("why any(Number) can not be replaced by any(Object)?", function() {
  var foo = jasmine.createSpy('foo');
  foo(12, function() {});
  expect(foo).toHaveBeenCalledWith(jasmine.any(Number), jasmine.any(Function));
  expect(foo).toHaveBeenCalledWith(jasmine.any(Object), jasmine.any(Function)); // will fail
  expect(foo).toHaveBeenCalledWith(2, jasmine.any(Object)); // will fail
  expect(foo).toHaveBeenCalledWith(jasmine.anything(), jasmine.anything());

  var A = function(){};
  var a = new A();
  expect(a).toEqual(jasmine.any(Object));
  expect(a).toEqual(jasmine.any(A));
});
```

原因：`jasmine`使用强类型进行验证匹配。对应的验证关系

* String -> any(String) / anything()
* Number -> any(Number) / anything()
* Boolean -> any(Boolean) / anything()
* null -> any(Object)
* Function -> any(Function) / anything()
* Object -> any(Object) / anything()

关于Javascript的类型，参考`w3schol`的官方解释：

> ECMAScript 有 5 种原始类型（primitive type），即 Undefined、Null、Boolean、Number 和 String。

> 对变量或值调用 typeof 运算符将返回下列值之一：

>> * undefined - 如果变量是 Undefined 类型的
>> * boolean - 如果变量是 Boolean 类型的
>> * number - 如果变量是 Number 类型的
>> * string - 如果变量是 String 类型的
>> * object - 如果变量是一种引用类型或 Null 类型的