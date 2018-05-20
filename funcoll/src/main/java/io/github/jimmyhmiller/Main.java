package io.github.jimmyhmiller;

import java.util.List;
import java.util.stream.Collectors;

public class Main {


    private static List<Character> toChars(String str) {
        return str.chars().mapToObj(e -> (char) e).collect(Collectors.toList());
    }


    public static void main(String[] args) {

        String str = "test";

        Sequence<Character> chars = new ListSeq<>(toChars(str));

        Sequence<Integer> list = ListSeq.range(5);
        System.out.println(
            list
                .map(n -> n + 2)
                .map(n -> n * 2)
                .filter(n -> n % 2 == 0));
        System.out.println(list.filter(i -> i%2 == 0));
        System.out.println(list.concat(ListSeq.range(5)));
        System.out.println(list.reduce(0, (Integer n, Integer i) -> n + 1));
        System.out.println(list.map((Integer n) -> n + 1).last());
    }
}
