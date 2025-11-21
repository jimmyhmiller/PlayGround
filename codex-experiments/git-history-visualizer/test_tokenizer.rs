// Test to understand how imara-diff tokenizes files
use imara_diff::sources::byte_lines_with_terminator;

fn main() {
    // Test case 1: File with trailing newline
    let data1 = b"line 1\nline 2\nline 3\n";
    let count1 = byte_lines_with_terminator(data1).count();
    println!("File with trailing newline: {} lines", count1);

    // Test case 2: File without trailing newline
    let data2 = b"line 1\nline 2\nline 3";
    let count2 = byte_lines_with_terminator(data2).count();
    println!("File without trailing newline: {} lines", count2);

    // Test case 3: Empty file
    let data3 = b"";
    let count3 = byte_lines_with_terminator(data3).count();
    println!("Empty file: {} lines", count3);

    // Test case 4: Single line with newline
    let data4 = b"line 1\n";
    let count4 = byte_lines_with_terminator(data4).count();
    println!("Single line with newline: {} lines", count4);

    // Test case 5: Single line without newline
    let data5 = b"line 1";
    let count5 = byte_lines_with_terminator(data5).count();
    println!("Single line without newline: {} lines", count5);
}
