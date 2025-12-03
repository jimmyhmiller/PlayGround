// Port of classes.ts

pub fn bg_color(color: &str) -> String {
    format!("ig-bg-{}", color)
}

pub fn text_color(color: &str) -> String {
    format!("ig-text-{}", color)
}
