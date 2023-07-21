use framework::{app, App, Canvas, Color, Rect, Size};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Event {
    kind: String,
    event: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EventViewer {
    size: Size,
    events: Vec<Event>,
    y_scroll_offset: f32,
}

struct Ui {}
enum Component {
    Pane(Size, Box<Component>),
    List(Vec<Component>),
    Container(Box<Component>),
    Text(String),
}

impl Component {
    fn draw(&self, canvas: &mut Canvas) {
        match self {
            Component::Pane(size, child) => {
                let background = Color::parse_hex("#353f38");

                let bounding_rect = Rect::new(0.0, 0.0, size.width, size.height);

                canvas.save();
                canvas.set_color(&background);
                canvas.clip_rect(bounding_rect);
                canvas.draw_rrect(bounding_rect, 20.0);

                canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));
                child.draw(canvas);
            }
            Component::List(children) => {
                for child in children.iter() {
                    child.draw(canvas);
                    // TODO: need to translate based on height of component
                    canvas.translate(0.0, 30.0);
                }
            }
            Component::Container(child) => {
                child.draw(canvas);
            }
            Component::Text(text) => {
                canvas.save();
                canvas.set_color(&Color::parse_hex("#ffffff"));
                canvas.draw_str(text, 40.0, 0.0);
                canvas.restore();
            }
        }
    }
}

impl Ui {
    fn new() -> Self {
        Self {}
    }

    fn pane(&self, size: Size, child: Component) -> Component {
        Component::Pane(size, Box::new(child))
    }

    fn list(&self, events: &[Event], f: impl Fn(&Self, &Event) -> Component) -> Component {
        Component::List(events.iter().map(|event| f(self, event)).collect())
    }

    fn container(&self, child: Component) -> Component {
        Component::Container(Box::new(child))
    }

    fn text(&self, text: &str) -> Component {
        Component::Text(text.to_string())
    }
}

impl App for EventViewer {
    type State = Self;

    fn init() -> Self {
        let me = Self {
            size: Size::default(),
            events: Vec::new(),
            y_scroll_offset: 0.0,
        };
        me.subscribe("*".to_string());
        me
    }

    fn draw(&mut self) {

        let mut canvas = Canvas::new();

        let ui = Ui::new();
        let ui = ui.pane(
            self.size,
            ui.list(&self.events, |ui, event|
                ui.container(ui.text(&event.kind)
            )),
        );
        ui.draw(&mut canvas);
    }

    fn on_click(&mut self, _x: f32, _y: f32) {}

    fn on_key(&mut self, _input: framework::KeyboardInput) {}

    // TODO: It would be nice to have scroll work for free
    // by default and only need to deal with overriding it.
    fn on_scroll(&mut self, _x: f64, y: f64) {
        self.y_scroll_offset += y as f32;
        if self.y_scroll_offset > 0.0 {
            self.y_scroll_offset = 0.0;
        }
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.size.width = width;
        self.size.height = height;
    }

    fn get_state(&self) -> Self::State {
        self.clone()
    }

    fn set_state(&mut self, state: Self::State) {
        *self = state;
    }

    fn on_event(&mut self, kind: String, event: String) {
        self.events.push(Event { kind, event });
    }
}

app!(EventViewer);
