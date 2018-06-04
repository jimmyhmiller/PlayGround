package io.github.jimmyhmiller.injector;

import java.util.function.Function;

public class Main {
    public static void main(String[] args) {


        Injector context = new Injector();
        DependsOn dTest = context
                .addSingleton(IDummy.class, new Dummy("TEST"))
                .addSingleton(Injector.class, context)
                .getInstance(DependsOn.class);

        System.out.println(dTest.helloGoodBye());
    }
}
