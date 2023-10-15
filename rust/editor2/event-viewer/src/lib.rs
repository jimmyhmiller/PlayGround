use framework::{app, App, Canvas, Position, Ui, WidgetData};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Event {
    kind: String,
    event: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EventViewer {
    widget_data: WidgetData,
    events: Vec<Event>,
    y_scroll_offset: f32,
    x_scroll_offset: f32,
}

// TODO: This is okay, but the scrolling should be smooth
// Why is the color syntax not working now? Is it because I saved?
// Or is it because I closed that panel?

impl App for EventViewer {
    type State = Self;

    fn init() -> Self {
        let me = Self {
            widget_data: WidgetData::default(),
            events: Vec::new(),
            y_scroll_offset: 0.0,
            x_scroll_offset: 0.0,
        };
        me.subscribe("*");
        me
    }

    fn draw(&mut self) {
        let mut canvas = Canvas::new();

        let ui = Ui::new();
        let ui = ui.pane(
            self.widget_data.size,
            (0.0, 0.0),
            ui.list(
                (0.0, self.y_scroll_offset),
                self.events.iter(),
                |ui, event| {
                    let line_offset_start = self.x_scroll_offset.abs() as usize / 16;
                    let line_length = self.widget_data.size.width as usize / 16;
                    let text = &format!("{}, {}", event.kind, event.event);
                    let text = text
                        .get(line_offset_start..(line_offset_start + line_length).min(text.len()))
                        .unwrap_or("");
                    ui.container(ui.text(text))
                },
            ),
        );
        ui.draw(&mut canvas);
    }

    fn on_click(&mut self, _x: f32, _y: f32) {
        if self.events.len() > 100 {
            self.events.drain(0..(self.events.len() - 100));
        }
    }

    fn on_key(&mut self, _input: framework::KeyboardInput) {}

    // TODO: It would be nice to have scroll work for free
    // by default and only need to deal with overriding it.
    fn on_scroll(&mut self, x: f64, y: f64) {
        self.y_scroll_offset += y as f32;
        if self.y_scroll_offset > 0.0 {
            self.y_scroll_offset = 0.0;
        }
        self.x_scroll_offset -= x as f32;
        if self.x_scroll_offset > 0.0 {
            self.x_scroll_offset = 0.0;
        }
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.widget_data.size.width = width;
        self.widget_data.size.height = height;
    }

    fn on_move(&mut self, x: f32, y: f32) {
        self.widget_data.position = Position { x, y };
    }

    fn get_state(&self) -> Self::State {
        self.clone()
    }

    fn set_state(&mut self, state: Self::State) {
        *self = state;
    }

    fn on_event(&mut self, kind: String, event: String) {
        self.events.push(Event { kind, event });
        if self.events.len() > 100 {
            self.events.drain(0..(self.events.len() - 100));
        }
    }
}

app!(EventViewer);
