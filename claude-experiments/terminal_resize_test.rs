use iocraft::prelude::*;

#[derive(Default, Props)]
struct TerminalResizeTestProps {}

#[component]
fn TerminalResizeTest(mut hooks: Hooks, _props: &TerminalResizeTestProps) -> impl Into<AnyElement<'static>> {
    let (width, height) = hooks.use_terminal_size();
    let mut counter = hooks.use_state(|| 0);
    
    // Create a future that increments the counter every second
    hooks.use_future(async move {
        smol::Timer::after(std::time::Duration::from_secs(1)).await;
        counter += 1;
    });

    // Generate tall content to test the smart clearing logic
    let tall_content = (0..height.saturating_sub(10))
        .map(|i| format!("Line {} - Terminal size: {}x{} - Counter: {}", i + 1, width, height, counter))
        .collect::<Vec<_>>();

    element! {
        View(flex_direction: FlexDirection::Column, width: 100pct, height: 100pct) {
            View(border_style: BorderStyle::Classic, padding: Padding::all(1)) {
                Text(content: format!("Terminal Resize Test"))
                Text(content: format!("Terminal dimensions: {} x {}", width, height))
                Text(content: format!("Counter: {}", counter))
                Text(content: "Try resizing the terminal window!")
            }
            
            View(flex_direction: FlexDirection::Column, flex_grow: 1.0) {
                #(tall_content.into_iter().map(|line| element! {
                    Text(content: line)
                }))
            }

            View(border_style: BorderStyle::Classic, padding: Padding::all(1)) {
                Text(content: "Bottom section - should remain visible after resize")
            }
        }
    }
}

fn main() {
    smol::block_on(element!(TerminalResizeTest).fullscreen_render_loop()).unwrap();
}