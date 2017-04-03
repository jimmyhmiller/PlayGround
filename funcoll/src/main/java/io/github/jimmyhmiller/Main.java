package io.github.jimmyhmiller;

public class Main {
    public static void main(String[] args) {
        Sequence<Integer> list = ListSeq.range(5);
        System.out.println(
            list
                .map(n -> n + 2)
                .map(n -> n * 2)
                .filter(n -> n % 2 == 0));
        System.out.println(list.filter(i -> i%2 == 0));
        System.out.println(list.concat(ListSeq.range(5)));
        System.out.println(list.reduce(0, (Integer n, Integer i) -> n + 1));
        System.out.println(list.map((Integer n) -> n + 1));
    }
}
