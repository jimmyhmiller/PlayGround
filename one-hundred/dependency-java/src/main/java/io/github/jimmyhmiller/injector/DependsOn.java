package io.github.jimmyhmiller.injector;

public class DependsOn {
    private final Dummy dummy;
    private final Dummy dummy1;

    public DependsOn(Dummy dummy, Dummy dummy1) {
        this.dummy = dummy;
        this.dummy1 = dummy1;
    }

    public String helloGoodBye() {
        return dummy.sayHello() + " Goodbye!!!" + dummy1.sayHello();
    }
}
