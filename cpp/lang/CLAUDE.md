# Project Information

This is a project creating a C-like language, but we are doing things slightly different. Like Lisp, we are having a "Reader" before we do conversion to ast. The goal here is in the early days we will use this reader layer as a way to desurgar things, eventually, this wil power an nice macro system.




## Building
This C++ project uses its own build system in build.sh

You can find all the available things using
```bash
./build.sh
```
which will print a nice helpful message for you.

## Running Tests

To run tests, use the following command:

```bash
./build.sh test
```


## Coding Standards

Comments **MUST** only be for explaining why we are doing something non-obvious. You should not leave comments like

```cpp
// parse let statement
... code that parses let statement
```

If you see comments like this as you are working, remove them.

### Formating
All code needs to be formatted with `build.sh fmt`