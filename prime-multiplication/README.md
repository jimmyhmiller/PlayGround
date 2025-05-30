# Prime Multiplication Table

A Clojure program that when given a number (n), prints a multiplication table of the first n prime numbers. Example output:

           2   3   5   7  11  13  15  17  19  21
       2   4   6  10  14  22  26  30  34  38  42
       3   6   9  15  21  33  39  45  51  57  63
       5  10  15  25  35  55  65  75  85  95 105
       7  14  21  35  49  77  91 105 119 133 147
      11  22  33  55  77 121 143 165 187 209 231
      13  26  39  65  91 143 169 195 221 247 273
      15  30  45  75 105 165 195 225 255 285 315
      17  34  51  85 119 187 221 255 289 323 357
      19  38  57  95 133 209 247 285 323 361 399
      21  42  63 105 147 231 273 315 357 399 441

## Running

If you have lein installed:

```bash
lein run
```

```bash
lein run 100 # First argument controls how many primes are printed
```

If you would prefer to run it in a docker container:

```bash
docker build -t prime-multi .
docker run prime-multi
```

```bash
docker run -it prime-multi lein run 100 # Little awkward to pass argument in docker
```

## Design Decisions

The project briefing did not state a particular language, but it's fairly clear that it was written with Ruby in mind. I decided to write this in Clojure because I am not looking to do Ruby in the future and I thought a Clojure version would better showcase my skillset.

I also decided to reinvent the wheel a bit. There is a [very simple example](https://github.com/richhickey/clojure-contrib/blob/40b960bba41ba02811ef0e2c632d721eb199649f/src/examples/clojure/clojure/contrib/pprint/examples/multiply.clj) of printing a multiplication table in Clojure already. `cl-format` is very powerful and while I certainly couldn't explain the options passed into it without looking them up, I did have a version of this program using that method. I chose not to use it because it seemed to make the problem too easy. The project briefing mentioned not using the "Prime class", and since I'm not using Ruby, that doesn't apply, but I took that to mean that using cl-format to print the whole entire table was not exactly in the spirit of the exercise.

Finally, I didn't demonstrate a TDD/BDD style of development. I think tests can be fantastic tools, but I generally am not a TDD advocate. Rather than practicing TDD, I decided to implement clojure.specs and leverage those to do some generative testing. Generative testing can really help provide greater coverage of a domain of values to ensure your code works properly across examples you may not have considered. There are certainly more properties I could have encoded into spec, but I find with Clojure's repl driven development, you can [catch bugs faster than TDD](http://blog.cognitect.com/blog/2017/6/5/repl-debugging-no-stacktrace-required).

## Speed considerations

The first thing that I do with any project is Google to find prior approaches to the problem. I found a number of people on github solving this exact problem. In most cases, they had a focus which I found a bit strange. When attempting to make this problem fast, many people worked to implement faster and faster prime methods. There was typically discussion about how a particularly fast method (the sieve of eratosthenes) didn't work here, followed by a dicussion of the various methods they tried. 

I find this strange because with my program (and theirs too, as far as I can tell), prime number generation is not the bottleneck. Because of the nature of multiplication tables, we are going to be dealing with formatting and printing n^2 values. Finding n primes will always take less time than this n^2 operation. (I also tested this to ensure that this was the case.)

Given that, I focused my attention on speeding up formatting the rows and values for printing. Clojure's rich standard library really came in handy here. A significant speed-up came from switching `map` to `pmap`. As the numbers get larger, the overhead of spinning up so many threads decreases, but I even saw speed-ups for lower values despite the overhead. The second boost in speed came from using `memoize`. In this case, `memoize` trades memory usuage for speed. Because all values in a multiplication table are repeated twice, this gave a nearly 2x speed-up.

### Limitations

Because of the n^2 nature of multiplication tables, this solution becomes incredibly slow as n grows. The speed-ups that were added did speed things up (around 7 times faster according to my rough calculations), but the computations are still incredibly expensive. Perhaps there is a way to get rid of this overhead, but since you actually do have to produce n^2 elements, I didn't see a way to do so.

Because of this computational complexity in my generative tests, I capped the numbers to make sure they stay below `Integer/MAX_VALUE`. I could have changed these to longs or bigints, but I don't think anyone would realistically run this program with such large numbers. (If they would, they are more patient than I am.)





