#!/usr/bin/env python3
import random
import subprocess
import sys

def generate_nested_sexp(depth=0, max_depth=4):
    """Generate a properly nested and indented s-expression"""
    if depth >= max_depth or random.random() < 0.3:
        # Terminal case: simple atom
        return random.choice(['foo', 'bar', 'baz', '42', ':keyword', 'x'])

    # Choose delimiter type
    delim = random.choice(['()', '[]', '{}'])
    open_delim, close_delim = delim[0], delim[1]

    # Generate children
    num_children = random.randint(1, 4)
    children = []
    indent = '  ' * (depth + 1)

    for _ in range(num_children):
        child = generate_nested_sexp(depth + 1, max_depth)
        if '\n' in child:
            # Multi-line child
            children.append(f'\n{indent}{child}')
        else:
            # Single-line child
            children.append(child)

    # Format based on complexity
    if all('\n' not in c for c in children) and len(children) <= 2:
        # Single line
        return f'{open_delim}{" ".join(children)}{close_delim}'
    else:
        # Multi-line
        formatted_children = []
        for child in children:
            if child.startswith('\n'):
                formatted_children.append(child)
            else:
                formatted_children.append(f'\n{indent}{child}')
        return f'{open_delim}{"".join(formatted_children)}\n{"  " * depth}{close_delim}'

def remove_random_closing_delims(sexp, num_to_remove=3):
    """Remove random closing delimiters"""
    lines = sexp.split('\n')
    result_lines = []
    removed_count = 0

    for line in lines:
        new_line = line
        # Try to remove closing delimiters from this line
        if removed_count < num_to_remove and any(c in line for c in ')]}'):
            for delim in random.sample([')' ,']', '}'], k=random.randint(1, 3)):
                if delim in new_line and removed_count < num_to_remove:
                    # Find last occurrence and remove it
                    idx = new_line.rfind(delim)
                    if idx != -1:
                        new_line = new_line[:idx] + new_line[idx+1:]
                        removed_count += 1
                        if removed_count >= num_to_remove:
                            break
        result_lines.append(new_line)

    return '\n'.join(result_lines)

def count_delimiters(text):
    """Count opening and closing delimiters"""
    counts = {
        '(': 0, ')': 0,
        '[': 0, ']': 0,
        '{': 0, '}': 0
    }

    in_string = False
    in_comment = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not in_comment:
            in_string = not in_string
            continue

        if ch == ';' and not in_string:
            in_comment = True

        if ch == '\n':
            in_comment = False

        if not in_string and not in_comment and ch in counts:
            counts[ch] += 1

    return counts

def test_parinfer(input_text, test_name):
    """Test parinfer balance on input"""
    # Write input to temp file
    with open('test_fuzz_input.lisp', 'w') as f:
        f.write(input_text)

    # Run paredit-like balance
    result = subprocess.run(
        ['./target/release/paredit-like', 'balance', 'test_fuzz_input.lisp', '--dry-run'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ {test_name}: Failed with error")
        print(f"  Error: {result.stderr}")
        return False

    output = result.stdout

    # Check that delimiters are balanced
    counts = count_delimiters(output)

    balanced = (
        counts['('] == counts[')'] and
        counts['['] == counts[']'] and
        counts['{'] == counts['}']
    )

    if balanced:
        print(f"✅ {test_name}: Balanced correctly")
        print(f"   Parens: {counts['(']}={counts[')']}, Brackets: {counts['[']}={counts[']']}, Braces: {counts['{']}={counts['}']}")
        return True
    else:
        print(f"❌ {test_name}: Not balanced!")
        print(f"   Parens: {counts['(']} vs {counts[')']}")
        print(f"   Brackets: {counts['[']} vs {counts[']']}")
        print(f"   Braces: {counts['{']} vs {counts['}']}")
        print(f"\nInput:\n{input_text}")
        print(f"\nOutput:\n{output}")
        return False

# Run tests
random.seed(42)  # For reproducibility
passed = 0
failed = 0

print("Testing parinfer with randomly generated nested s-expressions...\n")

for i in range(20):
    # Generate a well-formed s-expression
    sexp = generate_nested_sexp(depth=0, max_depth=random.randint(2, 5))

    # Test 1: Well-formed input should stay balanced
    if test_parinfer(sexp, f"Test {i+1}a: Well-formed"):
        passed += 1
    else:
        failed += 1

    # Test 2: Remove some closing delimiters and see if it balances
    broken = remove_random_closing_delims(sexp, random.randint(1, 3))
    if test_parinfer(broken, f"Test {i+1}b: Missing delimiters"):
        passed += 1
    else:
        failed += 1

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
print(f"Success rate: {100 * passed / (passed + failed):.1f}%")

sys.exit(0 if failed == 0 else 1)
