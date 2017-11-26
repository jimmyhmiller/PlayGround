# Phone Number

A simple http based api designed to store and retrieve phone numbers.

## Running

```bash
lein ring server-headless
```

By default the server will run on port 3000. If you want to run it on a different port you can run the following command.

```bash
lein ring server-headless <port>
```

If you'd rather just run the tests you can run:

```bash
lein test
```

### Example Requests

Everyone has their own favorite way to test http servers. I could possibly give them all. So I will show you two examples using my favorite to [httpie](https://httpie.org/).

#### Request - query for number

```bash
http ":3000/query?number=13196891881"
```

##### Reponse

```http
HTTP/1.1 200 OK
Content-Length: 64
Content-Type: application/json; charset=utf-8
Date: Sun, 26 Nov 2017 02:53:57 GMT
Server: Jetty(7.6.13.v20130916)

[
    {
        "context": "blah", 
        "name": "Bast Fazio", 
        "number": "+13196891881"
    }
]
```

#### Request - insert number

```bash
http post ":3000/number" number="+13196891881" name="testasdf" context="sasdfasdfsadfs"
```

##### Response

```http
HTTP/1.1 201 Created
Content-Length: 70
Content-Type: application/json; charset=utf-8
Date: Sun, 26 Nov 2017 02:52:04 GMT
Server: Jetty(7.6.13.v20130916)

{
    "context": "sasdfasdfsadfs", 
    "name": "testasdf", 
    "number": "+13196891881"
}
```

### Data Decisions

The sample data that I was given for this problem had a couple of issues. First of all, there are numbers that are not valid US numbers, there are  also number and context pair duplicates which goes against the invariants stated in the requirements. For the latter, I decided to just strip out any that were duplicates. There were only a handful of such duplicates, so keeping the invariants in place seemed more important.

As for the invalid US numbers, I decided to keep them in the dataset and I decided to allow new invalid numbers to be added. I made this decision for a few reasons. The prevalence of this invalid numbers factored into my decision to keep them, if I remember correctly there were 30,000+ invalid numbers. Also even valid US numbers don't guarentee that the number entered belongs to the user. If this number is important, verification via text or voice ought to be added in the future.

### Library Decisions

I decided to use compojure to build the api purely out of familiarity. The other library I considered was pedestal, which I do think could simplify some of the code, but I decided to stick with what I've mainly used in the past. I chose to use clojure.spec.alpha for this project mainly to act as a lightweight sanity check on the code base. I certainly could flesh out the spec definitions quite a bit more, but as they exist now, they have already prevented me from making a number of mistakes.

For testing I decided to keep it simple and use cognitect's new library transcriptor. It consists of one or more .repl files which some check statements scattered through out. It offers a really quick way to write tests, especially when working in an interactive environment like emacs+cider. To augment the tests, I included orchestra which augments clojure.spec.test to check :ret values when instrumenting functions.

That last library of not is googles phonelib. I initially used the [clojure wrapper](https://github.com/vlobanov/libphonenumber) for it, but found that it tried to do more than I needed it to and that caused it to be significantly slower. Instead I chose just to do some java interop. Using this library allowed me to not have to worry about formatting numbers myself, it seems to handle quite a few different formats with no issues.

### Limitations

I intentionally made this app only work with US numbers. The sample dataset only contained US numbers so I could accommodate all that was needed initially. Keeping this to US numbers simplified things when I didn't have a ton of time to work on it.

Due to time constraints I was not able break the code up into small commits as I would normally do. I only had random times to work on this problem and frequent interruptions, so the flow of writing it was not as smooth and I neglected a practice I normally follow. 

The code isn't tested or commented to the extent I would like. Again this is mainly do to time constraints and the lack of dedicated time to work on this. There is test coverage of the two endpoints including the different statuses they may return.

One last thing to note, start up time is fairly slow (around 20 seconds) mostly from parsing out the numbers. This is initially loading 500,000 phone numbers though and it never needs to do that again after startup, so it seems fairly acceptable.
