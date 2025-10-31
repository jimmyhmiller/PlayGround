// Tests using actual Pyret examples from the pyret-lang repository
use pyret_attempt::parse_program;

#[test]
fn test_parse_simple_if() {
    // From examples/concat-strings.arr
    let source = r#"
if not(false):
  print(string-append("Ahoy world!", "test"))
else:
  print("another thing")
end
    "#;

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse if statement: {:?}", result.err());
}

#[test]
fn test_parse_function_with_where() {
    // Simplified from examples/queue.arr
    let source = r#"
fun size(q):
  q.from-front.length() + q.from-back.length()
end
    "#;

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse function: {:?}", result.err());
}

#[test]
fn test_parse_data_simple() {
    // From examples/queue.arr
    let source = r#"
data Queue:
  | queue(from-front, from-back)
end
    "#;

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse data definition: {:?}", result.err());
}

#[test]
fn test_parse_data_with_methods() {
    // From examples/point.arr
    let source = r#"
data Color:
  | red
    with: method torgb(_): rgb(255, 0, 0) end
  | green
    with: method torgb(_): rgb(0, 255, 0) end
  | blue
    with: method torgb(_): rgb(0, 0, 255) end
  | rgb(r, g, b)
    with: method torgb(self): self end
sharing:
  method mix(self, other):
    fun avg(n1, n2): (n1 + n2) / 2 end
    rgb1 = self.torgb()
    rgb2 = other.torgb()
    rgb(
        avg(rgb1.r, rgb2.r),
        avg(rgb1.g, rgb2.g),
        avg(rgb1.b, rgb2.b)
      )
  end
end
    "#;

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse data with methods: {:?}", result.err());
}

#[test]
fn test_parse_cases_expression() {
    // From examples/queue.arr
    let source = r#"
fun dequeue(q):
  cases(List) q.from-back:
    | empty =>
      cases(List) q.from-front:
        | empty => raise("Dequeue on an empty queue")
        | link(_, _) =>
          new-from-back = q.from-front.reverse()
          new-queue = queue([list: ], new-from-back.rest)
          result = new-from-back.first
          { q: new-queue, v: result }
      end
    | link(f, r) => { q: queue(q.from-front, r), v: f }
  end
end
    "#;

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse cases expression: {:?}", result.err());
}

#[test]
fn test_parse_object_with_method() {
    // From examples/point.arr
    let source = r#"
point-methods = {
  method dist(self, other):
    ysquared = num-sqr(other.y - self.y)
    xsquared = num-sqr(other.x - self.x)
    num-sqrt(ysquared + xsquared)
  end
}
    "#;

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse object with method: {:?}", result.err());
}

#[test]
fn test_parse_check_block() {
    // From examples/point.arr
    let source = r#"
fun make-point(x, y):
  point-methods.{ x: x, y: y }
end

check:
  make-point(1, 1).dist(make-point(1, 3)) is 2
end
    "#;

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse check block: {:?}", result.err());
}

#[test]
fn test_parse_import() {
    // From examples/point.arr
    let source = r#"
import equality as E

fun test(): 5 end
    "#;

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse import: {:?}", result.err());
}

#[test]
fn test_parse_empty_program() {
    let source = "";

    let result = parse_program(source);
    assert!(result.is_ok(), "Failed to parse empty program: {:?}", result.err());
}
