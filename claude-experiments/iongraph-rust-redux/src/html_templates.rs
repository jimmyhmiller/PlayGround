// HTML template system for standalone HTML generation

pub struct HTMLTemplate {
    pub title: String,
    pub css: String,
    pub passes_sidebar_html: String,
    pub graphs_html: String,
    pub javascript: String,
}

impl HTMLTemplate {
    pub fn new(
        title: String,
        passes_sidebar_html: String,
        graphs_html: String,
        javascript: String,
    ) -> Self {
        HTMLTemplate {
            title,
            css: IONGRAPH_CSS.to_string(),
            passes_sidebar_html,
            graphs_html,
            javascript,
        }
    }

    pub fn render(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{}</title>
  <style>
{}
  </style>
</head>
<body>
  <div id="app">
    <div class="ig-sidebar">
{}
    </div>
    <div class="ig-viewport">
{}
    </div>
  </div>
  <script>
{}
  </script>
</body>
</html>"#,
            self.title, self.css, self.passes_sidebar_html, self.graphs_html, self.javascript
        )
    }
}

// Embedded CSS from TypeScript version
const IONGRAPH_CSS: &str = include_str!("../assets/iongraph.css");
