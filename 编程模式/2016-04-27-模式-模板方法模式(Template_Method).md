# 模式-原型模式（Prototype Pattern）和模板方法模式（Template Method）


## 原型模式

用原型实例指定创建对象的种类，并通过拷贝这些原型来创建新的对象。原型模式实现的关键在于实现Clone函数，对c++而言就是拷贝构造函数。


Prototype模式其实就是常说的"**虚拟构造函数**"一个实现,C++的实现机制中并没有支持这个特性, 但是通过不同派生类实现的Clone接口函数可以完成与"虚拟构造函数"同样的效果.举一个例子来解释这个模式的作用,假设有一家店铺是配钥匙的,他对外提供配制钥匙的服务(提供Clone接口函 数),你需要配什么钥匙它不知道只是提供这种服务,具体需要配什么钥匙只有到了真正看到钥匙的原型才能配好.也就是说,需要一个提供这个服务的对象,同时还需要一个原型(Prototype),不然不知道该配什么样的钥匙. 

### 例子

``` cpp
// 声明一个虚拟基类
class Prototype{
public:
    Prototype(){};
    virtual ~Prototype(){};
    virtual Prototype* clone() = 0;
};

class ConcreatePrototype: public Prototype{
public:
    ConcreatePrototype(){};
    ConcreatePrototype(const ConcreatePrototype& concreatePro){/* copy data here*/};
    virtual ~ConcreatePrototype(){};
    virtual Prototype* clone(){return new ConcreatePrototype(*this);};
};

void main(){
    Prototype* ptype = new ConcreatePrototype();
    Prototype* copyed_ptype = ptype->clone();
    delete ptype;
    delete copyed_ptype;
}
```



## 模板方法模式

定义了一个操作中的算法的骨架，而将部分步骤的实现在子类中完成。模板方法模式使得子类可以不改变一个算法的结构即可重定义该算法的某些特定步骤。

### 例子

```cpp
// 抽象类
class AbstractClass{
public:
    void TemplateMethod(){ // 总的方法
  		Operation1();
        Operation1();
        ConcreteOperation();
    }; 
    virtual void Operation1() = 0; // 由子类实现的方法
    virtual void Operation2() = 0;
    void ConcreteOperation(){}; // 抽象类实现方法
}

// 具体类
class ConcreteClass: public AbstractClass{
public:
    virtual void Operation1(){/*具体实现方法*/};
    virtual void Operation2(){/*具体实现方法*/};
}

void main(){
    AbstractClass* subject = new ConcreteClass();
    subject->TemplateMethod();
  
    delete subject;
}
```


## 参考

1. [c++ 设计模式之原型模式](http://blog.csdn.net/lbqbraveheart/article/details/7086883)
2. [C++设计模式-TemplateMethod模板方法模式](http://www.cnblogs.com/jiese/p/3180477.html)

