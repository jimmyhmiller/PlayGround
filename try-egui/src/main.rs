#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use egui::{RichText, containers::Frame, Color32, Visuals};

fn main() {





    let mut options = eframe::NativeOptions::default();
    // options.decorated = false;
    options.fullsize_content = true;

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|cc| {
            let my_app = MyApp::default();
            setup_custom_fonts(&cc.egui_ctx);
            Box::new(my_app)
        },
    ));
}

fn setup_custom_fonts(ctx: &egui::Context) {
    // Start with the default fonts (we will be adding to them rather than replacing them).
    let mut fonts = egui::FontDefinitions::default();


    // Install my own font (maybe supporting non-latin characters).
    // .ttf and .otf files supported.
    fonts.font_data.insert(
        "Inter".to_owned(),
        egui::FontData::from_static(include_bytes!("/Users/jimmyhmiller/Downloads/Inter-3.19/Inter Desktop/Inter-Regular.otf")),
    );

    // Put my font first (highest priority) for proportional text:
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "Inter".to_owned());

    // Put my font as last fallback for monospace:
    // fonts
    //     .families
    //     .entry(egui::FontFamily::Monospace)
    //     .or_default()
    //     .push("my_font".to_owned());

    // Tell egui to use these fonts:
    ctx.set_fonts(fonts);
    // let style = ctx.style().clone();
    // style.text_styles.
    let mut dark_mode = egui::Visuals::dark();
    dark_mode.override_text_color = Some(Color32::from_rgb(239,238,134));

    ctx.set_visuals(dark_mode);
    // ctx.set_style(style)
}



struct MyApp {
    name: String,
    age: u32,
}

impl MyApp {

}

impl Default for MyApp {

    fn default() -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
        }
    }
}



impl eframe::App for MyApp {



    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {


        let my_frame = Frame {
            inner_margin: egui::style::Margin { left: 40., right: 40., top: 40., bottom: 40. },
            outer_margin: egui::style::Margin { left: 0., right: 0., top: 0., bottom: 0. },
            rounding: egui::Rounding { nw: 0.0, ne: 0.0, sw: 0.0, se: 0.0 },
            shadow: eframe::epaint::Shadow { extrusion: 0.0, color: Color32::YELLOW },
            fill: Color32::from_rgb(64, 66, 96),
            stroke: egui::Stroke::new(0.0, Color32::from_rgb(157,252,203)),
        };


        egui::CentralPanel::default().frame(my_frame).show(ctx, |ui| {
            ui.heading(RichText::new("My Egui Application").size(60.0));
            ui.horizontal(|ui| {
                ui.label(RichText::new("Hello").size(40.0));
                ui.text_edit_singleline(&mut self.name);
            });
            ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            if ui.button("Click each year").clicked() {
                self.age += 1;
            }
            ui.label(format!("Hello '{}', age {}", self.name, self.age));
        });
    }
}
