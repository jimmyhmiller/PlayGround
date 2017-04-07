# OOP from the ground up

Any concept when taken for granted can lead to misunderstanding. Concepts we don't understand appear magical, making them either scary or compelling. In fact, typically the difference in attitude towards these magical objects is that of our peers. Despite the popular view that programming is an ever changing field, new ideas in the programming world must fight a hard fought battle. Often things which aren't familiar to the programmer who encounters them are met with suspicion or disdain. But not so with objects, or at least not anymore. Objects have captured the popular mindshare of developers. Objects are the bread and butter of programming, to the point where people are often confused at how languages with no objects can exist at all. We are going to peel back those covers and implement our own objects from the ground up.

## The Simplest Object

In order to begin, we must have a target in mind for our first object. A "hello world" of objects, from there we will move to more advanced features implementing them ourselves along the way.

```javascript
var person = {
	firstName: "Alonzo",
	lastName: "Church"
}

person.firstName // Alonzo
person.lastName // Church
```

This is our target, so in order to implement it we must understand it. The object above has two properties, firstName and lastName and it has some way to access those properties (the . operator). Now, our "object", because it isn't built into the language, is certainly not going to have as nice of a syntax as the one above, but we can definitely emulate the behavior of the object above.

```javascript
function person(property) {
	if (property === 'firstName') {
		return 'Alonzo';
	} else if (property === 'lastName') {
		return 'Church';
	}
}

person('firstName') // Alonzo
person('lastName') // Church

```

So here it is, our first "object". It might be hard to see exactly how our function here is an object, but with a little squinting we can see that it fulfills the exact same role as an object. The normal object has two properties which we can access, so does our "object". The only difference between them is the method of access. Our normal objects have their properties accessed through the dot operator, while our "objects" are through function application. In fact, a simple shift in language can show just how similar our "object" is to real objects. 

### Terminological 

SmallTalk is one the first OO languages; almost all of what is thought about as OO stems from it. Unfortunately we have lost a bit of SmallTalks terminology, terminology which would make things more clear. In the languages we are used to there are two ways to do things with objects, accessing properties (or fields) and invoking methods. With SmallTalk, there was just a single abstraction, message passing (This is called the "uniform access principle" and is the reason people often cite for Java getters/setters.) This "limitation" does not make SmallTalk any less capable. Everything you can do with objects in a modern language can be done in SmallTalk. Once we think about the dot operator as simple sending a message to our object, is our function application any different? We are simple sending the message as a string and our object is replying.

## More Advanced Objects

Our first object (I hope you can see that I am justified in remove those quotes) was fairly limited. In order to make a new person with a different name, we would have to go write a whole new function. But our original javascript object had the same problem, while simpler in syntax, we still how to write out the whole thing. Let's fix that.


```javascript

function createPerson(firstName, lastName) {
	return {
		firstName: firstName,
		lastName: lastName
	}
}

var person = createPerson('Alonzo', 'Church');
person.firstName // Alonzo
person.lastName // Church
```


Simple enough change, we just made a function that takes some parameters and returns our object. In fact, we can do the same for our object.

```javascript
function createPerson(firstName, lastName) {
	return function(property) {
		if (property === 'firstName') {
			return firstName;
		} else if (property === 'lastName') {
			return lastName;
		}
	}
}

var person = createPerson('Alonzo', 'Church');
person('firstName') // Alonzo
person('lastName') // Church

```

Since our object is just a function, we create a function that returns a function. Even with this "factory" function, our object continues to work just as it did before. But some of you may think, that's not a "real" object, "real" objects have methods. So let's add a method.

### Methods

```javascript
function createPerson(firstName, lastName) {
	return {
		firstName: firstName,
		lastName: lastName,
		fullName: function() {
			return firstName + " " + lastName;
		}
	}
var person = createPerson('Alonzo', 'Church');
person.fullName(); // Alonzo Church
```

Alright, there we are, an object with a method, this won't be too hard to recreate using our function technique.

```javascript 
function createPerson(firstName, lastName) {
	return function(property) {
		if (property === 'firstName') {
			return firstName;
		} else if (property === 'lastName') {
			return lastName;
		} else if (property === 'fullName') {
			return function() {
				return firstName + " " + lastName;
			}
		}
	}
}

var person = createPerson('Alonzo', 'Church');
person('fullName')(); // Alonzo Church
```

That was simple enough. A method is really just a function. So all we need to do is have our object return a function when you access a property. Then you can call that function. Again though, some people might be saying, this isn't a "real" object, "real" objects encapsulate state. 


### State

```javascript

function makeCounter() {
	return {
		value: 0,
		increment: function() {
			this.value += 1;
			return this.value;
		}
	}
}

var counter1 = makeCounter();
var counter2 = makeCounter();

counter1.increment() // 1
counter1.increment() // 2

counter2.increment() // 1
counter1.increment() // 3
```

Here we have an objects which encapsulates a bit of state. Each counter here keeps its own value. We can call increment on one counter, without affecting the other. This might seem a bit tricky to implement using our function-style objects, but it actually is no more complicated than any others.


```javascript
function makeCounter() {
	var value = 0;
	return function(property) {
		if (property === 'increment') {
			return function() {
				value += 1;
				return value;
			}
		}
	}
}

var counter1 = makeCounter();
var counter2 = makeCounter();

counter1('increment')() // 1
counter1('increment')() // 2

counter2('increment')() // 1
counter1('increment')()// 3
```

Our object with encapsulated state makes use of closures to hold state. In fact, in this version our state is actually further encapsulated because our value isn't publicly accessible. The only way anyone can get at the value is by sending the message 'increment'.

### More to come

This is of course a lot more to object oriented programming than what has been shown here. Most notably missing from the discussion is inheritance. Unfortunately addressing inheritance is a bit outside the scope of this article. Perhaps we will revisit it at some point in the future. But even with that feature left out, I hope some of the magic has been removed from objects. Objects aren't special. We don't need magic to make them, we just need simple functions.