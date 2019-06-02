# Meander for Practical Data Transformation

As Clojure programmers we love data. We believe that, at its core, programming is just data manipulation. To that end, Clojure provides fantastic data literals for wonderful immutable data structures. Moreover it provides a swiss-army knife of functions for manipulation of data. But as our data grows more complex, things become difficult, our beautiful declarative data transformation pipeline becomes a nested mess. We wind up, yet again, playing computer in our head.

In this tutorial we are slowly going to build up to understand how Meander can be used in Clojure to solve practical data transformation problems. We will start with simple examples and move to move complicated ones, hopefully choosing problems that reflect the sorts of menial data transformation tasks we all encounter in our day jobs. Let's start with some vanilla clojure code.

```clojure
(def person
  {:name "jimmy"
   :preferred-address
   {:address1 "123 street ave"
    :address2 "apt 2"
    :city "Townville"
    :state "IN"
    :zip "46203"}})

(defn reformat-preferred-address [person]
  (let [address (:preferred-address person)]
    {:address {:line1 (:address1 person)
               :line2 (:address2 person)}
     :city-info {:city (:city address)
                 :state (:state address)
                 :zipcode (:zip address)}}))
```

Here we have a pretty decent Clojure function that converts between two different address formats. This sort of code is fairly common when we need to convert from the data requirements of one system to another. Honestly, with this simple example, the code is fairly start forward. Our data requirements are simple and so our code isn't difficult. Let's look at how to accomplish this same task in Meander.

```clojure
(defn reformat-preferred-address [person]
  (m/match person
    {:preferred-address 
     {:address1 ?address1
      :address2 ?address2
      :city ?city
      :state ?state
      :zip ?zip}}
    
    {:address {:line1 ?address1
               :line2 ?address2}
     :city-info {:city ?city
                 :state ?state
                 :zipcode ?zip}}))
```

Here is code that does the same thing written with Meander. One obvious thing to note here is that the Meander version is much longer. Judging code based on number of lines is not something we are going to do. 

Let's explain what is going on. First we are using the Meander's `match` feature. `match` takes the thing that we are matching on (`person`), a pattern to try to match, and the output. Our pattern here is in the exact shape of the person map we are passed in. In order to extract out pieces of this map, we use logic variables (`?address1`, `?address2`, etc). Logic variables are just symbols that start with `?`. We can assign values in our data to any logic variables we'd like and then use those logic variables in our output. One thing I love about this simple Meander example, is that you can see the exact shape of the input immediately.

## Making Our Example Harder

This example while somewhat realistic is very limited. While I like the fact that Meander's match shows us the shape of our data, for simple examples like this, Clojure does pretty well. Let's make things harder.

```clojure
(def person
  {:name "jimmy"
   :preferred-address
   {:address1 "123 street ave"
    :address2 "apt 2"
    :city "Townville"
    :state "IN"
    :zip "46203"}
   :other-addresses 
   [{:address1 "432 street ave"
     :address2 "apt 7"
     :city "Cityvillage"
     :state "New York"
     :zip "12345"}
    {:address1 "534 street ave"
     :address2 "apt 5"
     :city "Township"
     :state "IN"
     :zip "46203"}]})
```

In the example above we left out somethings. A person has a preferred address, but they also have other addresses. We have a few different things we want to do with this data. First we want to find all the distinct zipcodes that a person has.

```clojure
(defn distinct-zip-codes [person]
  (let [preferred-address-zip (get-in person [:preferred-address :zip])
        other-zips (map :zip (:other-addresses person))]
    (distinct (cons preferred-address-zip other-zips))))
```

Here is some pretty straight forward Clojure code for doing exactly that. I'm sure some people could have minor quibbles about how this is written, but I doubt other solutions would be much different. One thing to note here is that we have lost a little bit the structure of our input data. We could maybe change that up a bit. Maybe using distructuring is the right approach? Regardless, this is a simple and reasonable Clojure function. Now, let's look at the Meander version.

```clojure
(defn distinct-zip-codes [person]
  (m/match person
    {:preferred-address {:zip !zips}
     :other-addresses [{:zip !zips} ...]}
    (distinct !zips)))
```

Here is the exact same function, but we've introduced two new concepts. The first are memory variables, in this case `!zip`. Memory variables start with `!` and remember all the values they match with. The next concept we introduced here is the zero or more operator (`…`). The zero or more operator says to repeat the pattern to its left zero or more times. In this case `{:zip !zips}`. Using these two, we can declaratively gather up all the zipcodes in this data structure.

### Minor Modifications

What happens if one of our zipcodes is nil? Well for both of our functions, nil gets returned in the output. That is probably not what we want. Let's fix that in both versions.

```clojure
(defn distinct-zip-codes [person]
  (let [preferred-address-zip (get-in person [:preferred-address :zip])
        other-zips (map :zip (:other-addresses person))]
    (filter some? (distinct (cons preferred-address-zip other-zips)))))
```

```clojure
(defn distinct-zip-codes [person]
  (m/match person
    {:preferred-address {:zip (pred some? !zips)}
     :other-addresses [{:zip (pred some? !zips)} ...]}
    (distinct !zips)))
```

These two functions aren't that different. In Meander we could have used `filter` in the exact same way if we wanted. But it's nice that we can set these conditions on the input, which is really more closely stating our intent. Here we used the `pred` operator which says to match something only if the predicate we give it returns true. Before we move on to more complex examples, let's consider one more modification. This time we want a distinct list of non-nil zips and cities output in a map like this `{:zips [] :cities []}`.

```clojure
(defn distinct-zips-and-cities [person]
  (let [preferred-address-zip (get-in person [:preferred-address :zip])
        preferred-address-city (get-in person [:preferred-address :city])
        other-zips (map :zip (:other-addresses person))
        other-cities (map :city (:other-addresses person))]
    {:zips (filter some? (distinct (cons preferred-address-zip other-zips)))
     :cities (filter some? (distinct (cons preferred-address-city other-cities)))}))
```

```clojure
(defn distinct-zips-and-cities [person]
  (m/match person
    {:preferred-address {:zip (pred some? !zips)
                         :city (pred some? !cities)}
     :other-addresses [{:zip (pred some? !zips)
                        :city (pred some? !cities)} ...]}
    {:zips (distinct !zips)
     :cities (distinct !cities)}))
```

With both of these examples, I extended them in the most obvious way I could think of. I think the Meander held up pretty well, but I wouldn't have written the plain Clojure function that way. Here's what I probably would have done instead.

```clojure
(defn distinct-zips-and-cities [person]
  (let [addresses (cons (:preferred-address person) 
                        (:other-addresses person))]
    {:zips (filter some? (distinct (map :zip addresses)))
     :cities (filter some? (distinct (map :city addresses)))}))
```

I think this is a pretty good function. But what I find interesting is that I needed to refactor to get here. It took me a little bit to think this way. Ideally, small changes to output should not require us to restructure our code. In this case the change is minor. But if we have to change our structure in such small cases, won't the we have to change it even more in larger cases?

## Searching with Meander

All our examples up until this point have had one answer. Yes, that answer might have been a collection, but there was only one way for our pattern to match. This isn't always the case. To see an example of that, let's write some functions using this data structure.

```clojure
(def people
  [{:name "jimmy"
    :addresses [{:address1 "123 street ave"
                 :address2 "apt 2"
                 :city "Townville"
                 :state "IN"
                 :zip "46203"
                 :preferred true}
                {:address1 "534 street ave",
                 :address2 "apt 5",
                 :city "Township",
                 :state "IN",
                 :zip "46203"
                 :preferred false}
                {:address1 "543 Other St",
                 :address2 "apt 50",
                 :city "Town",
                 :state "CA",
                 :zip "86753"
                 :preferred false}]}
   {:name "joel"
    :addresses [{:address1 "2026 park ave"
                 :address2 "apt 200"
                 :city "Town"
                 :state "CA"
                 :zip "86753"
                 :preferred true}]}])
```

I apologize for the amount of room this takes up on the screen, but real world examples are much larger. I want to try and make something that approaches realistic and in order to do that our input needs to be a bit bigger. Okay, so what we want do now is given a zip code, find all people that have an address with that zip code, and for each of the addresses that match that zip code, return a map of `{:name <name> :address <address>}`. So in this case if we asked for zip `86753` we should get the following response:

```clojure
({:name "jimmy",
  :address
  {:address1 "543 Other St",
   :address2 "apt 50",
   :city "Town",
   :state "CA",
   :zip "86753",
   :preferred false}}
 {:name "joel",
  :address
  {:address1 "2026 park ave",
   :address2 "apt 200",
   :city "Town",
   :state "CA",
   :zip "86753",
   :preferred true}})
```

Okay let's start with the vanilla Clojure example.

```clojure
(defn find-people-with-zip [people zip]
  (->> (for [person people]
         (for [address (:addresses person)
               :when (= (:zip address) zip)]
           {:name (:name person)
            :address address}))
       (mapcat identity)))
```

This code might not be very idiomatic. I almost never use `for` in actual code. But honestly this was the most succinct way I could think to write it. We could also have written something like this:

```clojure
(defn person-with-address-comb [person]
  (map (fn [address]
         {:name (:name person)
          :address address})
       (:addresses person)))

(defn find-people-with-zip [people zip]
  (->> people
       (mapcat person-with-address-comb)
       (filter (comp #{zip} :zip :address))))
```

Honestly, it seems like there is a better way I'm overlooking. But regardless I think any of these solutions will be a bit complicated. We've definitely lost the shape of the input data. We have some imperative stuff going on here. Let's contrast this with the Meander implementation.

```clojure
(defn find-people-with-zip [people zip]
  (m/search people
    (scan {:name ?name
           :addresses (scan {:zip ~zip :as ?address})})
    {:name ?name
     :address ?address}))
```

This is actually incredibly straight foward even if unfamiliar. We are now using `search` to find multiple answers. Also note `~zip`. The `~` here let's us splice in variables that are in scope. And finally, we can name our whole map using the `:as` pattern. This code reads like what we conceptually want to do, scan people's addresses looking for zips that match the one passed in. We do not have to think at all about how this code runs.

## One Final Example

For our final example of how Meander can be used to perform data manipulation, will show one feature of logic variables that we have left off so far. To do so we need some more complex data.

```clojure
(def data
  {:people 
   [{:name "jimmy" :id 1}
    {:name "joel" :id 2}
    {:name "tim" :id 3}]
   :addresses
   {1 [{:address1 "123 street ave"
        :address2 "apt 2"
        :city "Townville"
        :state "IN"
        :zip "46203"
        :preferred true}
       {:address1 "534 street ave",
        :address2 "apt 5",
        :city "Township",
        :state "IN",
        :zip "46203"
        :preferred false}]
    2 [{:address1 "2026 park ave"
        :address2 "apt 200"
        :city "Town"
        :state "CA"
        :zip "86753"
        :preferred true}]
    3 [{:address1 "1448 street st"
        :address2 "apt 1"
        :city "City"
        :state "WA"
        :zip "92456"
        :preferred true}]}
   :visits {1 [{:date "12-31-1900"
                :geo-location {:zip "46203"}}]
            2 [{:date "1-1-1970"
                :geo-location {:zip "12345"}}
               {:date "1-1-1970"
                :geo-location {:zip "86753"}}]
            3 [{:date "4-4-4444"
                :geo-location {:zip "54221"}}
               {:date "4-4-4444"
                :geo-location {:zip "92456"}}]}})
```

Here we have some much more realistic data than anything we've seen before. We have a map with three top level keys. These represent data we have gathered from various sources. The first `:people` is our collection of people with names and ids. The next are the `addresses` of these people, indexed by id for efficient lookup. And finally we have `:visits`. This represents the dates that the users visited our site, again index by user-id. 

Here's the mock scenario, we've seen supicious activity on our site and we aren't quite sure how to narrow it down. We are going to start our investigation by finding any users who had visits that were not in the same zip as their preferred address. Because of the nature of our application, we happen to know that it is typically used at the preferred location. Once we now the users affected we need to return their name, id, the date of access, and the zip code that didn't match.  But I want to show that despite this somewhat complicated scenario, we can easily express this using Meander. Before we get there, the Clojure implementation.

```clojure

(defn find-non-matching-visits [address visits]
  (filter (comp (complement #{(:zip address)}) :zip :geo-location) visits))

(defn find-bad-visits-for-person [addresses visits person]
  (let [preferred-address (first (filter :preferred addresses))
        non-matching (find-non-matching-visits preferred-address visits)]
    (map (fn [visit] {:name (:name person)
                      :id (:id person)
                      :zip (get-in visit [:geo-location :zip])
                      :date (:date visit)})
        non-matching)))

(defn find-potential-bad-visits [{:keys [addresses visits people]}]
  (mapcat (fn [{:keys [id] :as person}] 
            (find-bad-visits-for-person 
             (addresses id)
             (visits id) 
             person))
          people))
```

I really wanted to come up with a better implementation. `for` might have been useful here. If any reader has a better implementation, I'm happy to replace this one. But honestly, I think any implmentation is going to have the features that make this one less than desirable. Just look at how much of this code is about telling the computer what to do. Let's look at the Meander version now.

```clojure
(defn find-potential-bad-visits [data]
  (m/search data
    {:people (scan {:id ?id :name ?name})
     :addresses {?id (scan {:preferred true :zip ?zip})}
     :visits {?id (scan {:geo-location {:zip (and (not ?zip) ?bad-zip)}
                         :date ?date})}}
    {:name ?name
     :id ?id
     :zip ?bad-zip
     :date ?date}))
```

This is where Meander shines. `?id` is being used to join across data structures. We can now find an id in people and use that to index into other collections. This allows us to find out everything we need to know about a person easily. We can also search into any collection and match on data at any level. We don't need to rely on pulling things out into a higher scope by using let bindings, making helper functions to work on sub collections, or creating a series of transformations to get at the data we care about. Instead we declare exactly our data needs and the relationships that need to hold between them.

## Conclusion

I hope that this has been a good introduction to how Meander can be used for practical data transformation problems. In many of these examples, the vanilla clojure made for pretty good code. But as the data requirements become more complex, we need tools to handle these. While we may be able to accomplish any of these tasks, the udnerstanding of the structure of ou code becomes lost. Looking at the example above, we know so much about what the data coming in looks like. Our code mirrors precisely the shape of data we get in.

Now I do admit, my examples here are a bit contrived. But they are meant to be simple so we don't focus on the examples and instead focus on the code. In coming posts I will explore more directly various ways we can apply Meander for data transformation. Some ideas I have in mind are using Meander with honeysql to turn our data into sql, transforming a collection of data into hiccup for display as html, and using Meander to scrap the web. I'd also love to do more computer science examples. Using Meander to make little lisp interpretor, a CEK machine or basic arithmetic. And yet, Meander goes way beyond all of these things. Meander is about more than practical data manipulation. It is about a new away of programming, a new way of thinking about problems. Hopefully this introduction will help you to dive in and try it yourself.