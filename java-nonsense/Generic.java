import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.Arrays;

public class Generic {
    public static void main(String[] args) {
        List<Integer> myList = Arrays.asList(1,2,3);
        System.out.println(map((a) -> a + 2, myList));
    }

    public static <A,B> List<B> map(Function<A, B> f, List<A> list) {
        return list.stream().map(f).collect(Collectors.toList());
    }
}
// f :: (a, b) -> b
// f (a, b) = b