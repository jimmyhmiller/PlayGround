package io.github.jimmyhmiller.injector;

public class Main {
    public static void main(String[] args) {

        DependsOn d = new Injector()
                .getInstance(DependsOn.class);
        System.out.println(d.helloGoodBye());
    }
}
