let obj = {
    [Symbol.iterator]() {
        return 42;
    }
};
