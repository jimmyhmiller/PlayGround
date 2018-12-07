Just writing this to keep a note of what I've done. I want there to be no servers for me to manage, databases without servers is just a bit hard.

Initially I thought I might try S3 as a datastore. Sadly once I thought more about it, I realized that the eventual consistency was going to cause an issue. I considered aws serverless aurora, but the pricing model just doesn't make sense. You pay a ton for a little. I then thought about manta. It has the consistency guarantees I want. It has a lot of nice things about it, but 1) it isn't fast, 2) the libraries that support it don't like `ncc`, 3) it doesn't really allow efficient data fetching for my use cases.

Finally I'm trying faunadb. I had actually briefly looked at Fauna a long time ago. But it was very hard to decipher then. The documentation still isn't great, but I'm starting to see that the features it has might make this app great. `ncc` has no problem with it, the authentication system is a god send, and the pricing model is perfect.

My plan is to work on the schema, perhaps making a script that sets up scenarios I care about and allows me to see how they work with the schema and maybe time them.