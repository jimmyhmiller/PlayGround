class MyClass {
    #delete;
    #in;
    #instanceof;

    constructor() {
        this.#delete = 1;
        this.#in = 2;
        this.#instanceof = 3;
    }

    #if() {
        return this.#delete;
    }

    test() {
        this.#delete = 5;
        return this.#if();
    }
}
