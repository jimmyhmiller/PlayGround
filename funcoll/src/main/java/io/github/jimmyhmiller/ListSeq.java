package io.github.jimmyhmiller;


import java.util.ArrayList;
import java.util.List;

public class ListSeq<T> implements Sequence<T> {

    private final List<T> list;

    public ListSeq(List<T> list) {
        this.list = list;
    }

    public ListSeq(Sequence<T> seq) {
        this.list = seq.toList();
    }

    public static Sequence<Integer> range(Integer end) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < end; i++) {
            list.add(i);
        }
        return new ListSeq<>(list);
    }

    public T first() {
        return list.get(0);
    }

    public Sequence<T> rest() {
        if (list.size() == 1) {
            return this.empty();
        }
        return new ListSeq<>(list.subList(1, list.size()));
    }

    public Sequence<T> cons(T elem) {
        List<T> temp = new ArrayList<>();
        list.forEach(temp::add);
        temp.add(elem);
        return new ListSeq<>(temp);
    }

    public Sequence<T> empty() {
        return new ListSeq<>(new ArrayList<>());
    }

    public Boolean isEmpty() {
        return list.isEmpty();
    }

    public String toString() {
        return this.list.toString();
    }
}
