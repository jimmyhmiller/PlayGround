# Programs as Values - Fun with Free Monads

In this talk we will explore Free Monads. Depsite their scary name, Free Monads are not purely of academic interest, but can actually be useful in structuring real code. As our code grows in complexity, it is easy for our concerns to be mixed and our business logic to be unclear. Not only, does our code become messy, but it is riddled with side effects. Free Monads offer us a way out of this mess. 

We will see how to use free monads to turn programs into values. Using the Free Monad, we can embed  mini-programs into our application. These programs can then be interpretted in a variety of ways. This technique will allow us to separate our concerns, enabled to achieve things like dependency injection, and make our code more testable. 