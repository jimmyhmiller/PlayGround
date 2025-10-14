// @note: This is a test function with inline metadata
// @meta: {"tags":["test","demo"],"priority":5}
pub fn test_function() {
    println!("Testing inline metadata");
}

// @note: Another note without metadata
pub fn another_function() {
    println!("No inline metadata here");
}

// @note: Complex metadata example
// With multiline content
// @meta: {"tags":["complex","example"],"priority":8,"severity":"high","linked_issues":["ISSUE-123"]}
pub fn complex_example() {
    println!("Complex metadata");
}

fn main() {
    test_function();
    another_function();
    complex_example();
}
