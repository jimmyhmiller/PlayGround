# We Should be Able to Change our Languages

Programming languages are fixed artifacts. Defined by committee or that one really smart person. They are sacred artifacts whose contours are fixed for good reasons. The idea of changing your programming language, of customizing it for your own needs is seen as a drastic measure that should never be done. Or at least that's how people act. In a world where software is becoming increasingly flexible, moldable, editable, our programming languages resist this change.

Instead, we build incredibly complicated build systems. We create complex conventions that give us the semantics we wished our languages had. We reuse existing language constructs and give them different meanings. We form committees to advocate for changes to our languages, bike shedding endlessly, because the idea of just letting each individual codebase make its own decisions offends our sensibilities.

## Why We Resist

The ability for people to define their language to be the way they want is not new. It isn't difficult. But there has been a large backlash against it for decades. People avoid languages that offer this functionality. Even in languages that have it, some teams choose to completely ban it. But why exactly is this? I think the most straightforward version of the argument typically given is put quite well in [Raymond Chen's 2005 blog post A rant against flow control macros](https://devblogs.microsoft.com/oldnewthing/20050106-00/?p=36783). 

> When you create a flow-control macro, you’re modifying the language. When I fire up an editor on a file whose name ends in “.cpp” I expect that what I see will be C++ and not some strange dialect that strongly resembles C++ except in the places where it doesn’t.

Chen argues in the body of the post that these kinds of control flow macros obscure what one needs to understand when debugging. They hide the actual operations of the code. They replace the semantics expected of (in this case) cpp files with new semantics that are not known immediately to the person debugging the code.

The argument that only the authors of the macros will understand them and for everyone else they will be a hinderence has been echoed by many, even those who are big proponents of languages with this exact kind of power.

> Macros encourage people who are not good at language design to do something equivalent to language design, using tools that don't help, and with effects that are too powerful. This makes code unreadable to people joining later and for the authors after time has passed. Well designed macros are well documented, but this doesn't happen much. - Richard P. Gabriel

Gabriel isn't fully against macros. But is hesitant for people to write them. Believing there is some bar that must be met that most people don't meet.

Now of course there are other arguments that can be given. But I'll admit, I haven't really heard any that can't be cast in this same light. Reading code chock full of macros is confusing. You are reading a new programming language and we don't learn those super often. People hate macros for the same reason google made a language without generics, they make things more complicated.

## Do These Reasons Hold Up Anymore?

I'll be honest, I've never found the arguments against macros convincing. Control flow macros I've seen in the wild were always introduced to solve the very things Chen was worried about (I was in clojure not cpp though); they made sure at every sites resources were handled well. I didn't have to audit the code over and over at each call site, but once at a macro point. But I doubt most of my readers feel the same way. Defining our own language constructs has remained controversial and these arguments hold weight in the minds of many. So instead, I will try to convince you that the times have changed. That given the state of programming today, macros no longer pose as serious a problem even if we take the criticisms seriously.

### It Has Never Been Easier to Understand Code

Today, you aren't reliant on finding a local expert on some code, consulting the original author, asking on stackoverflow and hoping someone answers you, or spending a long time trying things out and reading docs, to understand a complicated piece of code. Instead, you can use an LLM to help you gain understanding in the ways that fit how you learn. For me, this often means custom visualizations, demos, asking increasingly specific questions. But sometimes a simple explanation and example are more than enough. Consider code I recent found that I didn't fully understand (not macro related).

```typescript
export type _ActionCreatorWithPreparedPayload<
  PA extends PrepareAction<any> | void,
  T extends string = string
> = PA extends PrepareAction<infer P>
  ? ActionCreatorWithPreparedPayload<
      Parameters<PA>,
      P,
      T,
      ReturnType<PA> extends {
        error: infer E
      }
        ? E
        : never,
      ReturnType<PA> extends {
        meta: infer M
      }
        ? M
        : never
    >
  : void
```

This is the kind of code where I can kind of piece together what each individual part might be doing but trying to understand the why and the what are difficult for me. I get lost in all the details. But with two questions to claude it was all cleared up for me. If you don't understand it, do the same. See how much easier this is than in the past?

### Understanding Each Bit of Code is Less Important Than It's Ever Been

There's quite a bit a controversy around the idea of commiting code that no one has read and understood. I've always worked in codebases large enough and old enough ([and ugly enough](https://jimmyhmiller.com/ugliest-beautiful-codebase)) that no one understands all the code written in them. In fact, I'll even contend that much of the code that was written wasn't understood when it was written. So this idea isn't super controversial to me. But I think people are starting to see what can be accomplished today without anyone understanding the code underneath. Full, useful applications can be built. Complicated projects can be launched.

Things have been changing drastically. There was once a point where it made sense to create suboptimal code if it meant that your team understood it better. Why? Because changes to the code provided more business value than the code itself being optimal, and if your team didn't understand it, no one could change things. But this is becoming less and less the case. If you can make code faster, better, even at the cost of readability, understandability, by you, but the AI can work with it perfectly fine, why wouldn't you?

 ## Sweetener, Macros for Typescript

So I think the time is ripe for macros. AI makes it trivial to write macros. AI makes it trivial to understand complex macros. Debugging code has become easier. And as I'll touch on later, I think AI written code might benefit greatly from macros. But there still is a problem. Language don't have them. At least not powerful ones. C style macros don't count. Rust has intentionally limited its macros in a way where they are not language constructs at all, but rather explicit, delimited things. Lisps are too weird for people. For a brief period in time [SweetJs](https://www.sweetjs.org/) was the perfect vehicle. But sadly, with the rewrite of it being abandoned, it no longer served that purpose. So, I have resurrected the idea of it, now aimed at Typescript called [Sweetener](https://github.com/sweetener-ts/sweetener). 

#### Our Own Constructs

So let's take a super brief tour of the kinds of things you can do with macros in sweetener. The pipe operator is a much debated, almost certainly dead proposal to javascript. Here is the good version (imo) of the pipe operator. The simple thread first version.

```typescript
import { (|>) } from "./macros.sts" for syntax;

function map...
function reduce...

export const total = [1, 2, 3]
|> map((n) => n * 2)
|> reduce((sum, n) => sum + n, 0);

export const longest = ["pipes", "read", "left", "to", "right"]
|> map((word) => word.length)
|> reduce((most, length) => Math.max(most, length), 0);

```

Rather than waiting for a committee for nearly a decade. We can create the operator ourselves. And what code do we need to accomplish this?

```typescript
export operator (|>):expr {
  fixity infix;
  associativity left;
  precedence 35;

  rule { $value:expr |> $function:ident $(. $member:ident)* ($($argument:expr),*) }
  expect "a function call, like `f(a)`, or a function name";
  => {
    $function $(. $member)*($value #if(present $argument) { , $($argument),* })
  }

  rule { $value:expr |> $function:ident $(. $member:ident)* }
  expect "a function call, like `f(a)`, or a function name";
  => {
    $function $(. $member)*($value)
  }
}

```

But simply adding a custom operator is not enough. You can add any constructor you want. Maybe your a fan of rust style enums aka algebriac datatypes

#### Algebraic Datatypes

```typescript
data Tree<T> = Leaf() | Node(left: Tree<T>, value: T, right: Tree<T>);

export const total = (tree: Tree<number>): number => {
  return match (tree) {
    Leaf() => 0;
    Node(left, value, right) => total(left) + value + total(right);
  };
}
```

Now, you can easily [define them yourself](https://sweetener-ts.github.io/#/play/matching) and customize things however you want. This version has pattern matching and exhaustiveness checking built-in.

#### Control Flow in TSX

```tsx

export const list = (items: readonly Item[]) => (
  <ul>
    {each (items as item, index)}
      <li key={item.id}>{index}: {item.name}</li>
    {end}
  </ul>
);
```

I know a ton of people have writing the map stuff in tsx. With sweetener, [you can define anything you want](https://sweetener-ts.github.io/#/play/jsx) and add it to your tsx without any pain or asking anyone else to support it.

### Why Care About Macros If I'm Not Reading the Code

It is perhaps the greatest irony that I am proposing things that make code look different when many people are no longer looking at code. Don't AIs just not care about all this higher level stuff? Can't we just have them write the lowest level code? Deal with all the boilerplate and they will just get it right?

I actually think macros might have an even greater benefit AI written code. In my experience, AI written code is quite good at introducing inconsistencies across repeated things. As codebases grow, they are not seeing the codebase as a whole. They do not follow the full conventions. What they do is instead copy code around them and match the constructs they see. A file full of macros for some nice error handling or for defining types in a nice standard way, or using the pipeline operator to make sure you have a nice disciple for how functions are defined, can help you tame your codebases implicitly. Macros encode larger patterns and automatically make sure AIs follow them.

## Macros are not Enough

I created sweetener in the hopes that it can give more people a chance to explore macros in a familiar context. But also, because macros in a setting like this are a fantastic way to explore language design. Rather than needing to create a compiler, we can play with the shape of a language feature that no existing language has and see if we can make something interesting. For example, [the morning paper covered a paper](https://blog.acolyer.org/2015/11/09/from-apis-to-languages-generalising-method-names/) that generalizes the idea of method names to allow regular expression like parts. We can now [prototype this and play with it](https://sweetener-ts.github.io/#/gist/ebf386fd9dc9868d15d597e2c2f85967).

But there is so much more we need to do with our languages. We have spent countless engineering hours building tooling, parsers, linters, type checkers, build tools all because our langauges don't give us the abiltiy to do what we need. Rather than letting us express the concepts we need to express as first class parts of our programs, rather than trusting us to use our powers well, to give us full expressivity, we have outsourced this to complicated external build tools, complicated external linting systems, testing frameworks with their own runners and conventions for running code.

And why? Because we didn't trust the average programmer to do it right. We didn't trust them to document it well. We didn't trust our future coworkers to understand what we had written. Code that was home grown was suspicious. Instead, we needed others to build our tooling for us. We needed to buy into a complicated external framework. We needed that division, because it was just too hard.

That's no longer the case. 

It has never been easier to build what you want to build. To imagine things and bring them to life. Its also never been easier to learn new things in programming. It has never been easier to migrate to a new language. To adopt new technology, or to make the customizations you need to if you language lets you. We are living with languages created when this wasn't reality. Languages afraid that their users will use them wrong. Languages that don't give us the power to do what we want to do. It is time that changed.