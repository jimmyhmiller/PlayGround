import java.util.HashSet;
import java.util.function.Function;
import java.util.*;


public class Validations {
  public static void main(String[] args) {
    Validation<String, String> s = Success.of("test");

    Validation<String, String> s1 = s
        .attempt(x -> new Failure<String, String>("I hate you"))
        .recover("otherTest")
        .attempt(x -> new Success<String, String>(x + "morestuff"));
        
    System.out.println(s1.toString());
  }
}

interface Validation<A, V> {
    <B> Validation<B, V> attempt(Function<A, Validation<B, V>> f);
    <B> Validation<B, V> map(Function<A, B> f);
    Set<V> validations();
    Validation<A, V> addValidations(Set<V> vs);
    Validation<A, V> recover(A a);
}


class Failure<A, V> implements Validation<A, V> {
    
    public final Set<V> vals;

    public Failure(Set<V> validations) {
        this.vals = validations;
    } 

    public Failure(V validation) {
        this.vals = new HashSet<V>(Arrays.asList(validation));
    } 
    
    public Set<V> validations() {
        return this.vals;
    }
    
    public Validation<A, V> addValidations(Set<V> vs) {
        this.vals.addAll(vs);
        return new Failure<A, V>(this.vals);
    }
    
    public <B> Validation<B, V> attempt(Function<A, Validation<B, V>> f) {
        return new Failure<B, V>(this.vals);
    }
    public <B> Validation<B, V> map(Function<A, B> f) {
        return new Failure<B, V>(this.vals);
    }
    
    public String toString() {
        return this.vals.toString();
    }
    
    public Validation<A, V> recover(A a) {
        return new Success<A, V>(a, this.vals);
    }
}

class Success<A, V> implements Validation<A, V> {
    
    public final A data;
    public final Set<V> vals;
    
    public static <B, U> Success<B, U> of(B b) {
        return new Success<B, U>(b, new HashSet<U>());
    } 
    
    public Success(A data, Set<V> validations) {
        this.data = data;
        this.vals = validations;
    }

    public Success(A data) {
        this.data = data;
        this.vals = new HashSet<>();
    }

    public Set<V> validations() {
        return this.vals;
    }
    
    public Validation<A, V> addValidations(Set<V> vs) {
        this.vals.addAll(vs);
        return new Success<A, V>(data, this.vals);
    }
    
    public <B> Validation<B, V> attempt(Function<A, Validation<B, V>> f) {
        Validation<B, V> newVal = f.apply(data);
        return newVal.addValidations(this.vals);
    }
    public <B> Validation<B, V> map(Function<A, B> f) {
        return new Success<B, V>(f.apply(this.data), this.vals);
    }
    
    public Validation<A, V> recover(A a) {
        return this;
    }
    
    public String toString() {
        return data.toString() + "\n" + this.vals.toString();
    }
}