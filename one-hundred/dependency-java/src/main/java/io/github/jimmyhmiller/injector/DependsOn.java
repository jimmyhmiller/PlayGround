package io.github.jimmyhmiller.injector;

public class DependsOn {
    private final IDummy dummy;
    private final IDummy dummy1;

    public DependsOn(IDummy dummy, Injector inj) {
        this.dummy = dummy;
        this.dummy1 = inj.getInstance(IDummy.class);
    }

    public String helloGoodBye() {
        return dummy.sayHello() + " Goodbye!!!" + dummy1.sayHello();
    }
}
