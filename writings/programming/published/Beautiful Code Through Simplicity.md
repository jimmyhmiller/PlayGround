# Beautiful Code Through Simplicity

Beautiful code is not about text. It isn't about the function it serves. It isn't line count. It isn't whitespace. It isn't how easy it was to write. Beautiful code is about one thing, structure. Simplicity of structure creates beautiful code, code that breaths, that hides no secrets; code that is readable, changeable, and comprehensible. If simplicity is to achieve these ends, it must be something a bit more than ease.

Ease in code writing is rather comfortable. It relaxes us, invigorates us, but ultimately it's deceptive. Ease is a measure of our skill; it shows us what we are familiar with, what makes sense to us. But as such, ease necessarily fades as our problem grows more complex. Ease will not create beautiful code; familiarity isn't always right. To obtain the beautiful code we seek, hard work is needed, not ease.

Simplicity is often used synonymously with ease. When we begin a tutorial for the new framework of the day, we will often think, "how simple!", and while "simple" does lend itself well to this usage, it is often necessary to hijack words, to purify them in order that our goal may be made clear. This process of turning an ordinary word, with its multifarious meanings, into a technical term may seem obnoxious to some, but once the process is through, speech becomes fluid, ideas can be built, and new areas explored.

## Complecting
Simplicity can best be understood through a rather unusual word, "complect". Complect literally means to "join by weaving". Complecting code is the process of taking two independent concerns and intertwining them together. To grasp this let's dive into some examples.

```javascript
function fetchData() {
	launchTheMissiles();
	return http.get("/data"); 
}
```
In this incredibly contrived example we can see a rather egregious example of complecting. What does fetching data have to do with launching missiles? Absolutely nothing. The caller of this function would be rather surprised to know that he launched missiles just by fetching some data. But perhaps we need a less contrived and more controversial example.

```javascript
let oldList = [1,2,3];
let newList = [];
for (let i = 0; i < oldList.length; i++) {
	newList.push(oldList[i] + 2);
} 
```

I'm sure the majority of people reading this see nothing wrong with code above. How is this code complecting anything? It is complecting data manipulation and time. To see this, ask yourself the following, what elements does newList contain? The answer depends on what time in the program it is. If we are before the loop none, in the middle of the loop it depends on what `i` equals, at the end of the loop [3,4,5]. Is this code easy to write? Of course but it lacks simplicity because it complects separate notions. Consider the following decomplected code.

```javascript
const oldList = [1,2,3];
const add2 = x => x+2;
const newList = oldList.map(add2);
```

Again let's ask the same question, what elements does newList contain? There is no condition here. It only ever contains one value [3,4,5]. Now imagine that oldList was huge. It contained millions of entries. Could we run our first version in parallel? No, encoded into to is the notion that we must iterate over the list sequentially. What about our second version? Of course we can. Map does not encode how the operation has to work, but just what it should do.

### What isn't being claimed

Unlike the first code I showed, this second one may not be as familiar. In fact, I wouldn't be surprised if some of you have never seen "map" before. So how can this code be more simple if fewer people are familiar with it? This is where we must remember that simplicity is not about familiarity. It is about keeping our concerns separate. Simplicity will not make it so everyone knows exactly what your code does. Its goal is to keep your code decomplected, because decomplecting allows composition.

## Composition

Complect and compose are opposites as far as programming goes. Where complecting mixes our concerns together, composition allows them to stay separate and be brought together in a straightforward fashion. Imagine that now instead of merely adding two to each element in our list we want to filter out all the evens and then add two. Our first example would change to this:

```javascript
let oldList = [1,2,3];
let newList = [];
for (let i = 0; i < oldList.length; i++) {
	if (i % 2 == 0) {
		newList.push(oldList[i] + 2);
	}
} 
```

Now as a developer I must follow in my head each step to determine what code is called. The if statement adds an additional branch my code can take making it that much harder to trace. Our second example will change as follows:

```javascript
const oldList = [1,2,3];
const add2 = x => x + 2;
const isEven = x => x % 2 == 0;
const newList = oldList.filter(isEven).map(add2);
```

Rather than including our changes into the body of some loop, we create functions that can be applied anywhere we'd like. But we can take our decomplecting one step further. Imagine now that our oldList is not longer a list, but a promise, how do our examples change? Let's start with the first example and see what perhaps seems like the most obvious way to change it.

```javascript
let oldList = getTheList();
let newList = [];w

oldList.then(function (list) {
	for (let i = 0; i < list.length; i++) {
		if (i % 2 == 0) {
			newList.push(list[i] + 2);
		}
	} 
});
```

Does this code work? Unfortunately no. We can see how our encoding of time caused us issues here. Again ask yourself the question, what elements does newList contain? Well, it depends if oldList has resolved or not.  If it hasn't newList will be empty. If it has then depending on what point in the for loop we are in, it has different values. We have introduced a race condition in our code. Of course we can fix this bug without transitioning our code fully, moving newList into the function and returning it will work, but as we will see in the second example, this sort of bug is not possible.

Now what about our second example?

```javascript
var oldList = getTheList();
var add2 = (x) => x+2;
var isEven = (x) => x%2 == 0;
var newList = oldList.then(list => 
	list.filter(isEven).map(add2)
);
```

This is the most obvious transformation. oldList is a promise, so obviously we can't directly filter on it. We must call then and apply our transformations. So now, newList is a promise which contains our list.  Our lack of complecting time and data transformation has paid off. 

# Conclusion

We could still yet take this code further down the path of decomplecting. Unfortunately, javascript doesn't cooperate much fully with decomplecting. But that isn't our concern for now. While these samples have been small, simplicity, in the sense of decomplecting, has already shown its benefits. As we move forward we will see more and more how our process of simplification can bring us closer and closer to our goal of beautiful code.







