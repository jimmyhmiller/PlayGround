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

The project briefing didn't state a particular language, but it is fairly clear that it was written with Ruby in mind. I decided to write this in Clojure, because I am not looking to do ruby in the future and I thought a Clojure version would better show case my skillset.

I also decided to reinvent the wheel a bit. There is already a [very simple example](https://github.com/richhickey/clojure-contrib/blob/40b960bba41ba02811ef0e2c632d721eb199649f/src/examples/clojure/clojure/contrib/pprint/examples/multiply.clj) of printing a multiplication table in clojure. `cl-format` is very powerful, while I certainly couldn't explain the options passed into it without looking them up, I did have a version of this program using that method. I chose not to use it, because it seemed to make the problem too easy. The project briefing mentioned not using the "Prime class", since I'm not using ruby, that doesn't apply, but I took that to mean that using cl-format to print the whole entire table was not exactly in the spirit of the exercise.

Finally, I didn't demonstrate a TDD/BDD style of development. I think tests can be fantastic tools, but I generally am not a TDD advocate. Rather than practicing TDD, I decided to implement clojure.specs and leverage those to do some generative testing. Generative testing can really help provide greater coverage of a domain of values to ensure your code works properly accross examples you may not have thought of. There are certainly more properties I could have encoded into spec, but I find with Clojure's repl driven developmen you can [catch bugs faster than TDD](http://blog.cognitect.com/blog/2017/6/5/repl-debugging-no-stacktrace-required).

## Speed considerations

First thing I do with any project is google to find prior approachs to a problem. I found a number of people on github solving this exact problem. In most cases, they had a focus which I found a bit strange. Many people when attempting to make this problem fast, worked to implement faster and faster prime methods. There was typically discussion about how a particularlly fast method (the sieve of eratosthenes) didn't work here. Then dicussion of the various methods they tried. 

I find this strange, because for my program (and as far as I can tell theirs too), prime number generation is not the bottle neck. Because of the nature of multiplication tables, we are going to be dealing with formatting and printing n^2 values. Finding n primes will always take less time than this n^2 operation. (I also tested this to ensure that this was the case.)

Given that, I focused my attention on speeding up formatting the rows and values for printing.





