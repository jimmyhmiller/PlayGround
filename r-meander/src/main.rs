use std::collections::VecDeque;


#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Pattern {
    Wildcard,
    LogicVariable(String),
    MemoryVariable(String),
    RepeatZeroOrMore(Box<Pattern>),
    Concat(Box<Pattern>, Box<Pattern>),
    StringConstant(String),
    IntegerConstant(i32),
}


fn b<T>(t : T) -> Box<T> {
    Box::new(t)
}


fn main() {

    let mut states = to_state_machine(
        Pattern::RepeatZeroOrMore(
            b(
                Pattern::Concat(
                    b(
                        Pattern::RepeatZeroOrMore(b(Pattern::IntegerConstant(1)))
                    ),
                    b(
                        Pattern::Concat(
                            b(
                                Pattern::RepeatZeroOrMore(b(Pattern::IntegerConstant(2)))
                            ),
                            b(
                                Pattern::Wildcard
                            )
                        )
                    )
                )
            )
        )
    );


    states.sort_by_key(|s| s.label);
    for state in states {
        println!("{:?}", state)
    }

}



type Label = i32;

#[derive(Debug)]
#[allow(dead_code)]
enum Op {
    NOOP,
    Return,
    Fail,
    SKIP,
    CheckString(String),
    CheckInteger(i32),
    Accumulate(String),
    LogicVariableAssign(String),
}

#[derive(Debug)]
struct State {
    label: Label,
    operation: Op,
    then_case: Label, // Maybe Option?
    else_case: Label // Maybe Option?
    // Or should the be an enum? There are states that can't fail and that is good to know
}

// Should make an interator for patterns
fn to_state_machine(pattern : Pattern) -> Vec<State> {
    let mut current_label = 0;
    let start = State { label: current_label, operation: Op::NOOP, then_case: current_label + 1, else_case: -1 };
    const END_STATE : i32 = std::i32::MAX;
    let end = State { label: END_STATE, operation: Op::Return, then_case: -1, else_case: -1};
    let mut states = vec!(start, end);
    current_label += 1;

    let mut queue = VecDeque::new();
    queue.push_front(pattern);

    let mut else_context = VecDeque::new();
    else_context.push_front(END_STATE);

    let mut next_context = VecDeque::new();

    next_context.push_front(current_label + 1);


    loop {
        if let Some(pattern) = queue.pop_front() {
            println!("{:?} {:?}", pattern, next_context);
            let next = next_context.pop_front().unwrap();
            let else_case = if queue.is_empty() { END_STATE } else { else_context.pop_front().unwrap_or(END_STATE) };

            match pattern {
                Pattern::Wildcard => {
                    states.push(State { label: current_label, operation: Op::SKIP, then_case: next, else_case: else_case });
                }
                // Really should go to current fail not end
                Pattern::IntegerConstant(i) => {
                    states.push(State { label: current_label, operation: Op::CheckInteger(i), then_case: next, else_case: else_case });
                }
                Pattern::StringConstant(s) => {
                    states.push(State { label: current_label, operation: Op::CheckString(s), then_case: next, else_case: else_case });
                }
                Pattern::MemoryVariable(name) => {
                    states.push(State { label: current_label, operation: Op::Accumulate(name), then_case: next, else_case: else_case });
                }
                Pattern::LogicVariable(name) => {
                    states.push(State { label: current_label, operation: Op::LogicVariableAssign(name), then_case: next, else_case: else_case });
                }
                Pattern::RepeatZeroOrMore(pattern) => {
                    queue.push_front(*pattern);
                    next_context.push_front(current_label);
                    if else_case != std::i32::MAX {
                        else_context.push_front(else_case);
                    } else {
                        else_context.push_front(next);
                    }
                    continue;
                }
                Pattern::Concat(p1, p2) => {
                    queue.push_front(*p2);
                    queue.push_front(*p1);
                    next_context.push_front(next);
                    next_context.push_front(current_label + 1);
                    if else_case != std::i32::MAX {
                        else_context.push_front(else_case + 1);
                        else_context.push_front(else_case + 1);
                    }
                    continue;
                }
                _ => break
            }

            current_label += 1;
            // Need to figure out how to handle last node.
            next_context.push_back(current_label + 1);
        } else {
            break;
        }
    }
    println!("{:?}", next_context);
    return states
}
