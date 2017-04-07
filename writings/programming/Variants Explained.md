# Variants Explained

Imagine you have the following request for a product you are building:

> Users should be able to login using the following methods:
>
> * Username and Password
> * Phone Number
> * Facebook Login

As you are building this application you have to think about how to represent these user credentials. It might look something like this:

```javascript

// Username and Password auth
userCredentials: {
	username: "test",
	password: "password"
}

// phoneNumber auth
userCredentials: {
	phoneNumber: "555-867-5309"
}

// Facebook Login
userCredentials: {
	token: "23rfaq234fy57rgsdfv3r==" 
}
```

There is a problem with the above way of representing this scenario. Imagine we needed to write a function that uses these user credentials. We have to make sure to handle every single case.

```javascript
function handleAuth(userCredentials) {
  if (userCredentials.username && userCredentials.password) {
	// do stuff with username login
  } else if (userCredentials.phoneNumber) {
      // do stuff with phone number login
  } else if (userCredentials.token) {
      // do stuff with facebook login
  } else {
      // handle unknown scenario
  }
}

```

This code made seem good as far as it goes, in fact, it even handles malformed data gracefully. Now imagine that our requirements change, we now need to handle third party username and password requests as well. We decide to model this in the obvious way.

```javascript
userCredentials: {
	username: "test",
	password: "password",
	thirdParty: "SomeOtherBusiness"
}
```


Unfortuantely now our code breaks, but not by throwing a nice error, it breaks subtly. We will try to use third party usernames and passwords for our own login system and since they have a username and password we will mistake them for first party logins.

In javascript, there aren't too many great solutions to this. One obvious one is to create a class for each of different userCredential type. Then for any function we want to implement on our different types we implement a method in that class. That is how a Java developer may have solved this problem. That approach has its upsides and downsides, but rather than dive into those, let's look at a different approach, one that isn't supported by javascript. Since it isn't supported by javascript we will have to choose some other language. But rather than choose an existing language, let's just make up our own and imagine what it might be like to solve this problem in it.


## Variants an Example

The essence of our requirements is that we need to support different methods of login. Use may login this way *or* that way *or* someother way. We need a way to represent **or** in our data model. Variants allow us to do exactly that. Let's first look at a simple example of a variant.

```haskell
data Color = Green | Red | Blue
```

Here we have a variant with three choices of colors. In our world, a color can only be green, red, or blue. No other colors are available to us. What we need to do now is write a function which returns true if it is passed the *best* color.

```javascript
fn bestColor {
	Green => true
	Red => false
	Blue => false
}
bestColor(Red)
// false

bestColor(Green)
//True
```

This function is rather straight-forward. We pattern match on the argument of the function to determine what was passed in. This allows us to express in a very concise way each case and what its output should be. Variants combined with pattern matching allow for very expressive, explicit code.

Simple variants like color are just like enums in other languages, but variants are much more useful when they can take arguments as well. 

```Haskell
data UserCredentials = FirstParty(username, password)
                     | Phone(phoneNumber)
                     | Facebook(token)
                     | ThirdParty(username, password, thirdParty)
```

Here we have our login problem fully specified. Each case is represented as a data type and because of that we can write a much less error prone functions for dealing with each case.

```Javascript
fn handleAuth {
  FirstParty(username, password) => // do stuff with username login
  Phone(phoneNumber ) => // do stuff with phone number login
  Facebook(token) => // do stuff with facebook login
  ThirdParty(username, password, thirdparty) => // do stuff with thirdParty login
  otherwise => // handle unknown scenario
}
```

Not only is our function less error prone, it is also much easier to understand. Variants allow our code to be self documenting. Each case is named and handle explicitly leading us to think precisely about each scenario. Since our imaginary language is dynamically typed, we do need to handle the `otherwise` case (imagine someone passed in a number instead), but if it were statically, we could be guarantee that nothing other than those variants would be passed to that function.

## Using Variants to Solve Problems

Variants not just limited to specific scenarios like the login above, they can be incredibly general and often more powerful because of it. Let's look at a few general variants that can be used to tackle common or difficult problems in programming.

### Nulls

Null (undefined as well) is one of the most frustating things to work with. Expressing nulls checks leads to verbose code, muddled with issues not neccessary for the problem at hand. Variants offer an alternative to nulls, called the Maybe type.

```haskell
data Maybe = Nothing | Something(thing)
```

The definition above may seem a bit strange if this is your first time encountering it. What it says is that there are two cases we need to consider, when we have nothing (the null case) and when we have something (the non-null case). We can use this by pattern matching.

```javascript
fn tryToGetUserId {
	Something(user) => Something(getId(user))
    Nothing => Nothing
}
```

The tryToGetUserId handles the case when we don't have a user id by pattern matching on `Nothing` and returning `Nothing`. If however we get something (a user) then we get the id of that user and return `Something` which contains a user.

As it stands, this isn't that much better than null, but when combined with simple functions, this variant because infinitely more useful.

```javascript
fn map {
  (f, Something(x)) => Something(f(x))
  (f, Nothing) => Nothing
}
```

Here we have map. You may be familiar with map with it comes to lists and if so, map for `Maybe` is very similar. As you can see from the function definition, map applies f only if we have `Something`, if not it returns `Nothing`. Using map we can rewrite our tryToGetUserId function.

```javascript
fn tryToGetUserId(maybeUser) {
  map(getId, maybeUser)
}
```

Using map extracts out all the pattern matching and does it for us. This same pattern can work for other variants. `map` is much more general than just lists.

```haskell
data Either = Error e | Success s
data List = Nil | Cons x tail
data Tree = Leaf | Node left x right
```

For each of these structures, there is a sensible map definition. `Either` allows us to handle errors and only apply the function if we are not in an error state, mapping over a `List` applies the function to each element, and mapping over a tree applies the function to each node, recursing through the tree.

## Conclusion

Variants are an extremely expressive way to state the assumptions behind our code. They force us to be expiclit and handle cases individual. Yet, they also give a means of abstraction, a point at which we can define common interfaces and ignored the particularities underneath. In future posts we will take up this notion in more depth, showing how `protocols` when combined with `variants` can bring our language even more power.