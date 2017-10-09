package io.github.jimmyhmiller;

public class Main {

    public static void main(String[] args) {
        Set<Integer> set = new EvenSet();

        System.out.println(set.contains(2));
        System.out.println(set.contains(4));
        System.out.println(set.contains(6));
        System.out.println(set.contains(7));
        System.out.println(set.insert(7).contains(7));
    }
}
