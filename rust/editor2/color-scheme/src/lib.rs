use framework::{App, app, Size};
use lsp_types::SemanticTokensLegend;
use serde::{Serialize, Deserialize};


#[derive(Clone, Debug, Serialize, Deserialize)]
struct ColorScheme {
    size: Size,
    token_legend: Option<SemanticTokensLegend>,
    colors: Vec<String>,
}

impl App for ColorScheme {
    type State = Self;

    fn init() -> Self {
        todo!()
    }

    fn draw(&mut self) {
        todo!()
    }

    fn on_click(&mut self, x: f32, y: f32) {
        todo!()
    }

    fn on_key(&mut self, input: framework::KeyboardInput) {
        // Need to be able to access clipboard
    }

    fn on_scroll(&mut self, x: f64, y: f64) {
        
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

}

app!(ColorScheme);