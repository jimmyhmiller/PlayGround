package io.github.jimmyhmiller.injector;

public class Dummy implements IDummy {

    public String name = "";

    public Dummy() {}

    public Dummy(String name) {
        this.name = name;
    }

    public String sayHello() {
        return "Hello!!! " + this.name;
    }
}
