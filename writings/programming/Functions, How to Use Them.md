# Functions, How to Use Them

This title may seem a bit too basic. Every programmer who has gone a bit beyond "Hello World" knows how to use functions, but in light of our previous post, [Side Effects, Complecting a la Carte](sideffects.md), we can see there is much more to using functions properly than simply understanding how to create and invoke them. Functions, when used properly, simplify the structure of our code, document our code, and give us code reuse far beyond what is traditionally taught.

## Purity Taken Seriously

In order to truly get the advantages functions are meant to provide, we must pay attention to the purity of our functions. Pure functions are those that only depend on their input and only produce an output. Let's explore a few examples of impure functions and then see how they complect our code and make it hard to reason about.

```javascript
const add = function (a, b) {
	return a + b;
};

const plus = function (a, b) {
	if (oppositeDay) {
		return a - b;
	}
	return a + b;
};
```

What do these functions do? Add things together. Well maybe, `plus` is an impure function, it depends on something outside its self that can change. Impure functions prevent us from reasoning consistently about our code, because we can never know what things affect them or what they will affect. Now, of course, `plus` as it is written here is not something anyone would do, but imagine the following more realistic example that might show up in an angular app.

```javascript
$scope.greetUser = function () {
	if ($scope.language === "es") {
		return "Hola " + $scope.user.name;
	}
	return "Hello " + $scope.user.name;
}
```

What does our function above do? Well, we can of course tell by reading the actual source of the function, but what about just its declaration? What data does our function depend on? The function itself doesn't tell us. It greets a user, but what user? The one on scope. What if we want to greet a different user? Well, we have to change scope. When we see "$scope.greetUser()" we can't know what it will do unless we know what values are on scope! This makes things very hard to reason about, we have to keep track of so many things in our head.

### Updating is impure

The impure functions we looked at above are impure because they change their behavior based on a variable that isn't passed in as an argument. But this isn't the only way a function can be impure. If a function updates something outside itself, it is impure. Examine the following example:

```Javascript
const isValid = function (user) {
    if (user.locked) {
	    accountLocked = true;
	    return false;
    }
    accountLocked = false;
    return true;
}
```

This seems innocent enough. For any user we pass in, we will always get the same output. But, since this function has a side effect we will get different behavior from what we expect. Imagine we want to use this function to only display a list of a users valid friends. Well, we might do something like the following:

```Javascript
var validFriends = user.friends.filter(isValid);
```

This will give us all and only our users valid friends. But imagine that the last friend in that list was invalid. Well, now we have a boolean that says our account it locked! But our account isn't locked! Complecting our data transformations and our updates can cause our functions to be un-reusable and potential introduce bugs in our code.

## But how do I do things?

Of course not all our functions can be pure (well, in javascript anyways). We do need to affect things in the world. So what should we do? The key to making code easier to reason about is by separating side effects from everything else we do. Imagine we split out our "isValid" function into two functions.

```Javascript
var isValid = function (user) {
	return !user.locked;
}

var setAccountLocked = function (bool) {
	accountLocked = bool;
	return bool;
}
```

Now, we are free to use "isValid" however we choose. It can have no ill effects from "improper" use of it. Pure functions are completely reusable.

### When to use Impure Functions

Splitting out our impure functions certainly makes our code more predictable, but it doesn't answer all of our questions. When and where should we use impure functions? Impure functions should be at the edges of our system. In other words, we should fetch data, do transformations on our data, and then output our data. Fetching and outputting are impure functions, while the core of our code is comprised entirely of pure functions.

This allows a few advantages. Firstly, testing is "all the rage", and our pure core allows us to much more easily test things. Pure functions don't depend on services. They don't depend on external resources. They only need what is passed to them, data. So testing doesn't involve 17 layers of mocks, but rather plain data. Secondly, our pure core is portable. If we decide to move our core out into a separate library, this change will be easy. Since each function depends on nothing outside of itself, transporting these functions is easy. Finally a pure core keeps our code neatly organize; when following the flow of our code we know precisely how things will happen, data will come in, transformations will happen, and data will come out.

# Conclusion

Pure functions are the first necessary step to unlocked the true potential of functions. In the next post we will explore more advanced uses of functions.  We will explore higher order functions, partially applied functions, and currying.
