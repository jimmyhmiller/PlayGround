package io.github.jimmyhmiller.objectalgebra;


import javax.json.JsonValue;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class Main<T> {



    public static <A> Builder<A> c(Object... args) {
        return (Collection<Converter<A>> c) -> c.create(args);
    }

    public static <A> A build(Function<Collection<Converter<A>>, Converter<A>> creator, Collection<Converter<A>> c) {
        return creator.apply(c).convert();
    }


    public static void main(String[] args) {

        Combination comb = new Combination();

        ListCollection<String> coll = new ListCollection<>();
        MapCollection<String, List<String>> map = new MapCollection<>(coll);
        JsonValueCollection json = new JsonValueCollection();


        Builder<List<String>> builder = c("test", "test");

        Builder b = c(
                "test1",
                    c("test", "test"),
                "test2",
                    c("test", "test")
        );

        Map<String, List<String>> s = build(b, map);
        JsonValue j = build(c("test", "test"), json);

        System.out.println(
                OptionalMonad.pure(2).map(x -> x+2)
        );

    }

}
