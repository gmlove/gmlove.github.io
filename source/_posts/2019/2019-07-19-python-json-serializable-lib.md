---
title: 如何实现一个优雅的Python的Json序列化库
categories:
- python
tags:
- python
- 序列化
- json
date: 2019-07-19 20:03:06
---

![python json serializable](/attaches/2019/2019-07-19-python-json-serializable-lib/serializable.png)

在Python的世界里，将一个对象以json格式进行序列化或反序列化一直是一个问题。Python标准库里面提供了json序列化的工具，我们可以简单的用`json.dumps`来将一个对象序列化。但是这种序列化仅支持python内置的基本类型，对于自定义的类，我们将得到`Object of type A is not JSON serializable`的错误。

有很多种方法可以用来支持这种序列化，[这里](https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable)有一个很长的关于这个问题的讨论。总结起来，基本上有两种还不错的思路：

1. 利用标准库的接口：从python标准json库中的`JSONDecoder`继承，然后自定义实现一个`default`方法用来自定义序列化过程
2. 利用第三方库实现：如`jsonpickle` `jsonweb` `json-tricks`等

利用标准库的接口的问题在于，我们需要对每一个自定义类都实现一个`JSONDecoder.default`接口，难以实现代码复用。

利用第三方库，对我们的代码倒是没有任何侵入性，特别是`jsonpickle`，由于它是基于`pickle`标准序列化库实现，可以实现像pickle一样序列化任何对象，一行代码都不需要修改。

但是我们观察这类第三方库的输出的时候，会发现所有的这些类库都会在输出的json中增加一个特殊的标明对象类型的属性。这是为什么呢？Python是一门动态类型的语言，我们无法在对象还没有开始构建的时候知道对象的某一属性的类型信息，为了对反序列化提供支持，看起来确实是不得不这么做。

有人可能觉得这也无可厚非，似乎不影响使用。但是在跨语言通信的时候，这就成为了一个比较麻烦的问题。比如我们有一个Python实现的API，客户端发送了一个json请求过来，我们想在统一的一个地方将json反序列化为我们Python代码的对象。由于客户端不知道服务器端的类型信息，json请求里面就没法加入这样的类型信息，这也就导致这样的类库在反序列化的时候遇到问题。

能不能有一个相对完美的实现呢？先看一下我们理想的json序列化库的需求：

1. 我们希望能简单的序列化任意自定义对象，只添加一行代码，或者不加入任何代码
2. 我们希望序列化的结果不加入任何非预期的属性
3. 我们希望能按照指定的类型进行反序列化，能自动处理嵌套的自定义类，只需要自定义类提供非常简单的支持，或者不需要提供任何支持
4. 我们希望反序列化的时候能很好的处理属性不存在的情况，以便在我们加入某一属性的时候，可以设置默认值，使得旧版本的序列化结果可以正确的反序列化出来

如果有一个json库能支持上面的四点，那就基本是比较好用的库了。下面我们来尝试实现一下这个类库。

对于我们想要实现的几个需求，我们可以建立下面这样的测试来表达我们所期望的库的API设计：

```python
class SerializableModelTest(unittest.TestCase):

    def test_model_serializable(self):

        class A(SerializableModel):

            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b if b is not None else B(0)

            @property
            def id(self):
                return self.a

            def _deserialize_prop(self, name, deserialized):
                if name == 'b':
                    self.b = B.deserialize(deserialized)
                    return
                super()._deserialize_prop(name, deserialized)

        class B(SerializableModel):

            def __init__(self, b):
                super().__init__()
                self.b = b

        self.assertEqual(json.dumps({'a': 1, 'b': {'b': 2}, 'long_attr': None}), A(1, B(2)).serialize())
        self.assertEqual(json.dumps({'a': 1, 'b': None}), A(1, None).serialize())

        self.assertEqual(A(1, B(2)), A.deserialize(json.dumps({'a': 1, 'b': {'b': 2}})))
        self.assertEqual(A(1, None), A.deserialize(json.dumps({'a': 1, 'b': None})))
        self.assertEqual(A(1, B(0)), A.deserialize(json.dumps({'a': 1})))
```

这里我们希望通过继承的方式来添加支持，这将在反序列化的时候提供一个好处。因为有了它我们就可以直接使用`A.deserialize`方法来反序列化，而不需要提供任何其他的反序列化函数参数，比如这样`json.deserialize(serialized_str, A)`。

同时为了验证我们的框架不会将`@property`属性序列化或者反序列化，我们特意在类`A`中添加了这样一个属性。

由于在反序列化的时候，框架是无法知道某一个对象属性的类型信息，比如测试中的`A.b`，为了能正确的反序列化，我们需要提供一点简单的支持，这里我们在类`A`中覆盖实现了一个父类的方法`_deserialize_prop`对属性`b`的反序列化提供支持。

当我们要反序列化一个之前版本的序列化结果时，我们希望能正确的反序列化并使用我们提供的默认值作为最终的反序列化值。这在属性`A.b`的测试中得到了体现。

（一个好的测试应该一次只验证一个方面，上面的测试是为了简洁起见写在了一起，而且也有很多边界的情况并没有覆盖。此测试只是作为示例使用。）

如果能有一个类可以让上面的测试通过，相信那个类就是我们所需要的类了。这样的类可以实现为如下：

```python
class ModelBase:

    @staticmethod
    def is_normal_prop(obj, key):
        is_prop = isinstance(getattr(type(obj), key, None), property)
        is_constant = re.match('^[A-Z_0-9]+$', key)
        return not (key.startswith('__') or callable(getattr(obj, key)) or is_prop or is_constant)

    @staticmethod
    def is_basic_type(value):
        return value is None or type(value) in [int, float, str, list, tuple, bool, dict]

    def _serialize_prop(self, name):
        value = getattr(self, name)
        if isinstance(value, (tuple, list)):
            try:
                json.dumps(value)
                return value
            except Exception:
                return [v._as_dict() for v in value]
        return value

    def _as_dict(self):
        keys = dir(self)
        props = {}
        for key in keys:
            if not ModelBase.is_normal_prop(self, key):
                continue
            value = self._serialize_prop(key)
            if not (ModelBase.is_basic_type(value) or isinstance(value, ModelBase)):
                raise Exception('unkown value to serialize to dict: key={}, value={}'.format(key, value))
            props[key] = value if self.is_basic_type(value) else value._as_dict()
        return props

    def _short_prop(self, name):
        value = getattr(self, name)
        if isinstance(value, (tuple, list)):
            try:
                json.dumps(value)
                return value
            except Exception:
                return [v._as_short_dict() for v in value]
        return value

    def _as_short_dict(self):
        keys = dir(self)
        props = {}
        for key in keys:
            if not ModelBase.is_normal_prop(self, key):
                continue
            value = self._short_prop(key)
            if not (ModelBase.is_basic_type(value) or isinstance(value, ModelBase)):
                raise Exception('unkown value to serialize to short dict: key={}, value={}'.format(key, value))
            props[key] = value if self.is_basic_type(value) else value._as_short_dict()
        return props

    def serialize(self):
        return json.dumps(self._as_dict(), ensure_ascii=False)

    def _deserialize_prop(self, name, deserialized):
        setattr(self, name, deserialized)

    @classmethod
    def deserialize(cls, json_encoded):
        if json_encoded is None:
            return None

        import inspect
        args = inspect.getfullargspec(cls)
        args_without_self = args.args[1:]
        obj = cls(*([None] * len(args_without_self)))

        data = json.loads(json_encoded, encoding='utf8') if type(json_encoded) is str else json_encoded
        keys = dir(obj)
        for key in keys:
            if not ModelBase.is_normal_prop(obj, key):
                continue
            if key in data:
                obj._deserialize_prop(key, data[key])
        return obj

    def __str__(self):
        return self.serialize()

    def _prop_eq(self, name, value, value_other):
        return value == value_other

    def __eq__(self, other):
        if other is None or other.__class__ is not self.__class__:
            return False

        keys = dir(self)
        for key in keys:
            if not ModelBase.is_normal_prop(self, key):
                continue
            value, value_other = getattr(self, key), getattr(other, key)
            if not (ModelBase.is_basic_type(value) or isinstance(value, ModelBase)):
                raise Exception('unsupported value to compare: key={}, value={}'.format(key, value))
            if value is None and value_other is None:
                continue
            if (value is None and value_other is not None) or (value is not None and value_other is None):
                return False
            if not self._prop_eq(key, value, value_other):
                return False

        return True

    def short_repr(self):
        return json.dumps(self._as_short_dict(), ensure_ascii=False)
```

为了更进一步提供支持，我们将最终的类命名为`ModelBase`，因为通常我们要序列化或反序列化的对象都是我们需要特殊对待的对象，且我们通常称其为模型，我们一般也会将其放在一个单独`models`模块中。

作为一个模型的基类，我们还添加了一些常用的特性，比如：

1. 支持标准的格式化接口`__str__`，这样我们在使用`'{}'.format(a)`的时候，就可以得到一个更易于理解的输出
2. 提供了一个缩短的序列化方式，在我们有时候不想直接输出某一个特别长的属性的时候很有用
3. 提供了基于属性值的比较方法
4. 自定义类的属性可以为基础的Python类型，或者由基础Python类型构成的`list` `tuple` `dict`

在使用这个类的时候，当然也是有一些限制的，主要的限制如下：

1. 当某一属性为自定义类的类型的时候，需要子类覆盖实现`_deserialize_prop`方法为反序列化过程提供支持
2. 当某一属性为由自定义类构成的一个`list` `tuple` `dict`复杂对象时，需要子类覆盖实现`_deserialize_prop`方法为反序列化过程提供支持
3. 简单属性必须为python内置的基础类型，比如如果某一属性的类型为`numpy.float64`，序列化反序列化将不能正常工作

虽然有上述限制，但是这正好要求我们在做模型设计的时候保持克制，不要将某一个对象设计得过于复杂。比如如果有属性为`dict`类型，我们可以将这个`dict`抽象为另一个自定义类型，然后用类型嵌套的方式来实现。

到这里这个基类就差不多可以支撑我们日常的开发需要了。当然对于这个简单的实现还有可能有其他的需求或者问题，大家如有发现，欢迎留言交流。
