# Conceptual Overview

## Entities and Types

Everything in the database is an **entity**. An entity is a thing with an identity and a set of fields. You give entities structure by defining **types**.

```js
db.defineType("User", {
  name:  { type: "string", required: true },
  email: { type: "string", unique: true },
  age:   { type: "i64" },
  bio:   { type: "string" },
})
```

A type says what fields an entity can have, what kind of value each field holds, and any constraints. The available value types are `string`, `i64`, `f64`, `bool`, and `bytes`.

Once a type is defined, you can create entities of that type:

```js
let alice = await db.table("User").insert({
  name: "Alice",
  age: 30,
  email: "alice@example.com",
})
```

And get them back:

```js
let user = await db.table("User").get(alice.id).run()
// => { id: 1, name: "Alice", age: 30, email: "alice@example.com" }
```

Fields that aren't `required` can be omitted. They simply won't be present on the entity. There are no null values — a field either exists or it doesn't.

---

## Relationships

Entities relate to each other through **refs**. A ref is a field that points at another entity.

```js
db.defineType("Post", {
  title:  { type: "string", required: true },
  body:   { type: "string" },
  author: { type: "ref", target: "User", required: true },
})
```

When you create a Post, you pass the author's entity ID:

```js
await db.table("Post").insert({
  title: "Hello World",
  body: "My first post",
  author: alice.id,
})
```

Refs are how you model all relationships:

```js
// One-to-many: a comment belongs to a post and a user
db.defineType("Comment", {
  body:   { type: "string", required: true },
  author: { type: "ref", target: "User", required: true },
  post:   { type: "ref", target: "Post", required: true },
})

// Self-referential: a user follows other users
db.defineType("Follow", {
  follower: { type: "ref", target: "User", required: true },
  followed: { type: "ref", target: "User", required: true },
})
```

### Cardinality

By default, a field holds one value. A field declared with `many: true` holds a **set** of values — zero or more, with no duplicates.

```js
db.defineType("Post", {
  title:  { type: "string", required: true },
  body:   { type: "string" },
  author: { type: "ref", target: "User", required: true },
  tags:   { type: "ref", target: "Tag", many: true },
})

db.defineType("Tag", {
  name: { type: "string", required: true, unique: true },
})
```

A Post can have zero, one, or many tags. No join table needed.

```js
let rust = await db.table("Tag").insert({ name: "rust" })
let db_  = await db.table("Tag").insert({ name: "databases" })

await db.table("Post").insert({
  title: "Building a Datalog DB",
  author: alice.id,
  tags: [rust.id, db_.id],
})
```

When you read the entity back, many-valued fields come back as arrays:

```js
let post = await db.table("Post").get(5).run()
// => { id: 5, title: "Building a Datalog DB", author: 1, tags: [10, 11] }
```

`many` works on any type, not just refs:

```js
db.defineType("User", {
  name:    { type: "string", required: true },
  emails:  { type: "string", many: true },
  scores:  { type: "i64", many: true },
})

await db.table("User").insert({
  name: "Alice",
  emails: ["alice@work.com", "alice@home.com"],
  scores: [100, 95, 88],
})
```

### Updating Many-Valued Fields

For a cardinality-one field, an update replaces the value. For a many-valued field, you have finer control:

```js
// Replace the entire set
await db.table("Post").get(5).update({
  tags: [rust.id, newTag.id],
})

// Add to the set
await db.table("Post").get(5).add({
  tags: [anotherTag.id],
})

// Remove from the set
await db.table("Post").get(5).remove({
  tags: [rust.id],
})
```

`add` asserts new values. `remove` retracts specific values. `update` on a many-valued field replaces the whole set (retracts everything, then asserts the new values).

### Filtering on Many-Valued Fields

A filter on a many-valued field matches if **any** value in the set satisfies the predicate:

```js
// Posts tagged with "rust"
let rustPosts = await db.table("Post")
  .filter(field("tags").contains(rust.id))
  .run()

// Users with any score above 90
let highScorers = await db.table("User")
  .filter(field("scores").contains(field.gt(90)))
  .run()
```

In datalog queries, a variable binding on a many-valued field produces one result row per value — it naturally fans out:

```js
// "Find all tag names for each post title"
let results = await db.query({
  find: ["?title", "?tagName"],
  where: [
    { bind: "?p", type: "Post", title: "?title", tags: "?t" },
    { bind: "?t", type: "Tag", name: "?tagName" },
  ]
})
// => [["Building a Datalog DB", "rust"], ["Building a Datalog DB", "databases"]]
```

This is the natural semantics — `?t` binds to each tag ref in turn, and for each binding the rest of the query executes.

---

## Enums: Sum Types

Sometimes a field doesn't hold a single kind of value — it holds one of several possible shapes. An **enum** defines a set of named variants, each with its own fields.

```js
db.defineEnum("Shape", {
  Circle: { radius: { type: "f64", required: true } },
  Rect:   { w: { type: "f64", required: true }, h: { type: "f64", required: true } },
  Point:  {},
})
```

A `Shape` is always exactly one of `Circle`, `Rect`, or `Point`. `Point` is a **unit variant** — it has no fields, it's just a tag.

You use an enum as a field type:

```js
db.defineType("Drawing", {
  label: { type: "string", required: true },
  shape: { type: "enum", target: "Shape", required: true },
})
```

Creating entities with enum fields:

```js
// Data variant — object with variant name as key
await db.table("Drawing").insert({
  label: "my circle",
  shape: { Circle: { radius: 5.0 } },
})

// Unit variant — just the variant name as a string
await db.table("Drawing").insert({
  label: "origin",
  shape: "Point",
})
```

Reading back an entity with an enum field:

```js
let drawing = await db.table("Drawing").get(10).run()
// => { id: 10, label: "my circle", shape: { Circle: { radius: 5.0 } } }

let point = await db.table("Drawing").get(11).run()
// => { id: 11, label: "origin", shape: "Point" }
```

Enums are validated on write. You can't use a variant name that doesn't exist, pass the wrong field types for a variant, or omit a required variant field.

### Changing Variants

When you update an enum field to a different variant, the old variant's fields are cleaned up automatically. You can't accidentally have both Circle and Rect data on the same entity.

```js
// Currently a Circle with radius 5.0
await db.table("Drawing").get(10).update({
  shape: { Rect: { w: 10.0, h: 20.0 } },
})

// Now it's a Rect. The radius is gone.
let d = await db.table("Drawing").get(10).run()
// => { id: 10, label: "my circle", shape: { Rect: { w: 10.0, h: 20.0 } } }
```

### Real-World Enum Examples

```js
db.defineEnum("PaymentMethod", {
  CreditCard: {
    last4:  { type: "string", required: true },
    brand:  { type: "string", required: true },
    expiry: { type: "string", required: true },
  },
  BankTransfer: {
    routingNumber: { type: "string", required: true },
    accountLast4:  { type: "string", required: true },
  },
  Crypto: {
    wallet:  { type: "string", required: true },
    network: { type: "string", required: true },
  },
})

db.defineEnum("OrderStatus", {
  Pending:    {},
  Processing: {},
  Shipped:    { carrier: { type: "string", required: true }, tracking: { type: "string" } },
  Delivered:  { deliveredAt: { type: "i64", required: true } },
  Cancelled:  { reason: { type: "string" } },
  Refunded:   { amount: { type: "i64", required: true }, reason: { type: "string" } },
})
```

---

## Querying

### All Entities

```js
let users = await db.table("User").run()
```

### Get by ID

```js
let user = await db.table("User").get(42).run()
```

### Filtering

Filter by exact field values:

```js
let alices = await db.table("User")
  .filter({ name: "Alice" })
  .run()
```

Filter with comparisons:

```js
let adults = await db.table("User")
  .filter(field("age").gt(18))
  .run()

let youngAdults = await db.table("User")
  .filter(field("age").gte(18))
  .filter(field("age").lt(30))
  .run()
```

Multiple `.filter()` calls are AND'd together. Available predicates: `eq`, `gt`, `lt`, `gte`, `lte`, `ne`.

### Filtering on Enums

Match by variant:

```js
let circles = await db.table("Drawing")
  .filter(field("shape").isVariant("Circle"))
  .run()
```

Filter on a variant's fields with dot notation:

```js
let bigCircles = await db.table("Drawing")
  .filter(field("shape").isVariant("Circle"))
  .filter(field("shape.radius").gt(10.0))
  .run()

let shipped = await db.table("Order")
  .filter(field("status").isVariant("Shipped"))
  .filter(field("status.carrier").eq("FedEx"))
  .run()
```

`shape.radius` means "the `radius` field inside the current variant of `shape`." If the entity's shape isn't a Circle (and has no `radius`), the filter doesn't match — no error.

### Sorting, Pagination, Projection

```js
let page = await db.table("User")
  .orderBy("name")
  .skip(20)
  .limit(10)
  .run()

let descending = await db.table("User")
  .orderBy("age", "desc")
  .run()

// Only return specific fields
let names = await db.table("User")
  .select(["name", "email"])
  .run()
// => [{ id: 1, name: "Alice", email: "alice@example.com" }, ...]
```

### Selecting Nested Fields

Without `.select()`, ref fields come back as raw IDs. `.select()` lets you follow refs and pick exactly what you want at each level.

```js
// Just IDs — no select
let posts = await db.table("Post").run()
// => [{ id: 5, title: "Hello", body: "...", author: 1, tags: [10, 11] }]

// Follow the author ref, pick specific fields
let posts = await db.table("Post")
  .select(["title", { author: ["name", "email"] }])
  .run()
// => [{ id: 5, title: "Hello",
//       author: { id: 1, name: "Alice", email: "alice@work.com" } }]

// Follow a many-valued ref
let posts = await db.table("Post")
  .select(["title", { tags: ["name"] }])
  .run()
// => [{ id: 5, title: "Hello",
//       tags: [{ id: 10, name: "rust" }, { id: 11, name: "databases" }] }]
```

Strings in the array are field names. Objects follow refs — the key is the ref field, the value is another selection for the target entity.

This nests arbitrarily deep:

```js
let comments = await db.table("Comment")
  .select([
    "body",
    { author: ["name"] },
    { post: ["title", { author: ["name"] }] },
  ])
  .run()
// => [{ id: 20, body: "Great post!",
//       author: { id: 2, name: "Bob" },
//       post: { id: 5, title: "Hello",
//               author: { id: 1, name: "Alice" } } }]
```

Use `"*"` to include all fields at a level:

```js
// All post fields + expanded author with all their fields
let posts = await db.table("Post")
  .select(["*", { author: ["*"] }])
  .run()

// Just expand the author, keep everything else as default
let posts = await db.table("Post")
  .select(["*", { author: ["name"] }])
  .run()
```

Enum fields are values, not refs — they always come back as a complete unit. If you include `"shape"` in a select, you get the whole enum value. There's no reaching into variant fields through select (that's what filter's dot notation is for).

```js
db.table("Drawing")
  .select(["label", "shape"])
  .run()
// => [{ id: 10, label: "my circle", shape: { Circle: { radius: 5.0 } } }]
```

### Count

```js
let total = await db.table("User").count().run()

let adultCount = await db.table("User")
  .filter(field("age").gte(18))
  .count()
  .run()
```

---

## Joins and Reverse Lookups

`.select()` handles the forward direction — following refs from the entity you're querying. For the reverse direction, filter by ID or use datalog.

```js
// "Find all Posts by Alice" — the ref lives on Post, filter by it
let alicePosts = await db.table("Post")
  .filter({ author: alice.id })
  .run()

// "Find all Posts tagged with rust" — reverse lookup on a many-valued field
let rustPosts = await db.table("Post")
  .filter(field("tags").contains(rust.id))
  .run()
```

### Multi-Entity Queries (Datalog)

For complex queries that span multiple entity types with shared variables, use the datalog query form:

```js
// "Find the names and titles of users who have published posts"
let results = await db.query({
  find: ["?name", "?title"],
  where: [
    { bind: "?u", type: "User", name: "?name" },
    { bind: "?p", type: "Post", author: "?u", title: "?title", published: true },
  ]
})
// => [["Alice", "Hello World"], ["Alice", "Second Post"]]
```

Anything starting with `?` is a **variable**. When the same variable appears in multiple places, those values must unify — that's the join. Here `?u` binds to User entities in the first clause and is used as the Post's author in the second clause.

```js
// "Find users who follow each other (mutual follows)"
let mutuals = await db.query({
  find: ["?a_name", "?b_name"],
  where: [
    { bind: "?f1", type: "Follow", follower: "?a", followed: "?b" },
    { bind: "?f2", type: "Follow", follower: "?b", followed: "?a" },
    { bind: "?a", type: "User", name: "?a_name" },
    { bind: "?b", type: "User", name: "?b_name" },
  ]
})
```

Many-valued fields and datalog work naturally together. Since each value in a many-valued field produces a separate binding, you get the join-table behavior without the join table:

```js
// "Find all tags shared between two posts"
let shared = await db.query({
  find: ["?title1", "?title2", "?tagName"],
  where: [
    { bind: "?p1", type: "Post", title: "?title1", tags: "?t" },
    { bind: "?p2", type: "Post", title: "?title2", tags: "?t" },
    { bind: "?t",  type: "Tag",  name: "?tagName" },
  ]
})
```

You can also match on enum variants in datalog queries:

```js
// "Find the labels and radii of all Circle drawings"
let circles = await db.query({
  find: ["?label", "?r"],
  where: [
    { bind: "?d", type: "Drawing", label: "?label",
      shape: { match: "Circle", radius: "?r" } }
  ]
})
```

---

## Updating and Retracting

### Partial Updates

Updates only touch the fields you specify. Everything else stays as-is.

```js
await db.table("User").get(alice.id).update({
  age: 31,
  bio: "Software engineer",
})
// name and email are untouched
```

Setting a field to the same value it already has is a no-op.

### Retracting Fields

Remove fields from an entity entirely:

```js
await db.table("User").get(alice.id).retract(["bio", "email"])
```

After retraction, those fields simply don't exist on the entity anymore. They're not null — they're absent.

---

## Time Travel

The database keeps complete history. Every write records what changed and when. You can query any historical point.

```js
let tx1 = await db.table("User").insert({ name: "Alice", age: 30 })
let tx2 = await db.table("User").get(alice.id).update({ age: 31 })

// Query the present
await db.table("User").get(alice.id).run()
// => { id: 1, name: "Alice", age: 31 }

// Query the past
await db.table("User").asOf(tx1.txId).get(alice.id).run()
// => { id: 1, name: "Alice", age: 30 }
```

`asOf` works with any query:

```js
// How many users existed at transaction 100?
await db.table("User").asOf(100).count().run()

// What orders were shipped at that point?
await db.table("Order")
  .asOf(100)
  .filter(field("status").isVariant("Shipped"))
  .run()

// Datalog queries support it too
await db.query({
  find: ["?name"],
  where: [{ bind: "?u", type: "User", name: "?name" }],
  asOf: 100,
})
```

Time travel is read-only. You can't modify the past.
