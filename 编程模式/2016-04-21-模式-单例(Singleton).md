# 模式-单例模式（Singleton）


## 模式编程中的单例模式。

<!-- more -->


``` cpp
class Singleton{
private:
    Singleton(){};
public:
    static Singleton& GetInstance(){
        static Singleton instance;
        return instance;
    }
};
```

## 调用方法：

``` cpp
Singleton::GetInstance().method();
```

## 错误用法：

``` cpp
Singleton singleton = Singleton::GetInstance();
```

## 错误原因：
赋值时调用了默认的复制构造函数，生成的是新的实例。

当多线程调用时，要注意线程安全。
