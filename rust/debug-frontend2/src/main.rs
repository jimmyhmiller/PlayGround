use gpui::*;

#[derive(IntoElement)]
struct Button {
    child: Option<AnyElement>,
}

impl Button {
    fn new() -> Self {
        Self { child: None }
    }
}

impl RenderOnce for Button {
    fn render(self, cx: &mut WindowContext) -> impl IntoElement {
        let button = div()
            .bg(rgb(0x2e7d32))
            .text_color(rgb(0xffffff))
            .text_xl()
            .hover(|style| style.bg(rgb(0x1b5e20)))
            .rounded(px(4.0))
            .flex()
            .p(px(4.0))
            .child(if let Some(child) = self.child {
                child
            } else {
                div().into_any_element()
            });

        button
    }
}

struct Counter {
    count: usize,
}

impl Counter {
    fn new() -> Self {
        Self { count: 0 }
    }
}

impl Render for Counter {
    fn render(&mut self, cx: &mut ViewContext<Self>) -> impl IntoElement {
        div()
            .flex()
            .size_full()
            .justify_center()
            .items_center()
            .text_xl()
            .bg(rgb(0xaa0000))
            .on_mouse_down(MouseButton::Left, cx.listener(move |this, event, cx| {
                println!("Clicked!");
            }))
            .text_color(rgb(0xffffff))
            .child(format!("Count: {}", self.count))
    }
}

impl ParentElement for Button {
    fn extend(&mut self, elements: impl Iterator<Item = AnyElement>) {
        let mut elements = elements;
        self.child = elements.next();
        if self.child.is_none() || elements.next().is_some() {
            panic!("Button can only have one child");
        }
    }
}

struct HelloWorld {
    text: SharedString,
}

impl Render for HelloWorld {
    fn render(&mut self, _cx: &mut ViewContext<Self>) -> impl IntoElement {
        div()
            .flex()
            .size_full()
            .justify_center()
            .items_center()
            .text_xl()
            .text_color(rgb(0xffffff))
            .child(format!("Hello, {}!", &self.text))
    }
}

struct Container {}

impl Render for Container {
    fn render(&mut self, cx: &mut ViewContext<Self>) -> impl IntoElement {
        div()
            .flex()
            .size_full()
            .justify_center()
            .items_center()
            .child(cx.new_view(|cx|Counter::new()))
    }
}
// I have no idea how this stuff is supposed to work

fn main() {
    App::new().run(|cx: &mut AppContext| {
        cx.open_window(WindowOptions::default(), |cx| {
            cx.new_view(|cx| Container {})
        });
    });
}
