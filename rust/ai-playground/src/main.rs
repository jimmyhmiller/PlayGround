use three_d::*;

#[derive(Debug, Clone)]
pub struct DrawingScript {
    pub name: String,
    pub commands: Vec<DrawCommand>,
}

#[derive(Debug, Clone)]
pub struct DrawingArea {
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub color: [f32; 4], // RGBA outline color for visual identification
    pub visible: bool,
}

impl DrawingArea {
    pub fn new(name: String, x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            name,
            x,
            y,
            width,
            height,
            color: [0.5, 0.5, 0.5, 0.3], // Default gray with transparency
            visible: true,
        }
    }
    
    pub fn contains_point(&self, point_x: f32, point_y: f32) -> bool {
        point_x >= self.x && point_x <= self.x + self.width &&
        point_y >= self.y && point_y <= self.y + self.height
    }
    
    pub fn to_relative_coords(&self, world_x: f32, world_y: f32) -> (f32, f32) {
        ((world_x - self.x), (world_y - self.y))
    }
    
    pub fn to_world_coords(&self, local_x: f32, local_y: f32) -> (f32, f32) {
        (self.x + local_x, self.y + local_y)
    }
}

#[derive(Debug, Clone)]
pub struct ScriptAssignment {
    pub script_name: String,
    pub area_name: String,
}

pub struct ScriptManager {
    pub scripts: Vec<DrawingScript>,
    pub areas: Vec<DrawingArea>,
    pub assignments: Vec<ScriptAssignment>,
    pub script_sources: std::collections::HashMap<String, String>, // Store original script text for time updates
}

impl ScriptManager {
    pub fn new() -> Self {
        Self {
            scripts: Vec::new(),
            areas: Vec::new(),
            assignments: Vec::new(),
            script_sources: std::collections::HashMap::new(),
        }
    }
    
    pub fn add_area(&mut self, area: DrawingArea) {
        self.areas.push(area);
    }
    
    pub fn create_area(&mut self, name: String, x: f32, y: f32, width: f32, height: f32) {
        let area = DrawingArea::new(name, x, y, width, height);
        self.add_area(area);
    }
    
    pub fn remove_area(&mut self, name: &str) {
        // Remove area
        self.areas.retain(|a| a.name != name);
        // Remove any assignments to this area
        self.assignments.retain(|a| a.area_name != name);
    }
    
    pub fn update_area(&mut self, name: &str, x: f32, y: f32, width: f32, height: f32) {
        if let Some(area) = self.areas.iter_mut().find(|a| a.name == name) {
            area.x = x;
            area.y = y;
            area.width = width;
            area.height = height;
        }
    }
    
    pub fn get_area_mut(&mut self, name: &str) -> Option<&mut DrawingArea> {
        self.areas.iter_mut().find(|a| a.name == name)
    }
    
    pub fn parse_and_add_script(&mut self, input: &str, time: f32) {
        let script = DrawingParser::parse_script_with_time(input, time);
        self.script_sources.insert(script.name.clone(), input.to_string());
        self.update_script(script);
    }
    
    pub fn update_scripts_with_time(&mut self, time: f32) {
        let sources: Vec<(String, String)> = self.script_sources.iter()
            .map(|(name, source)| (name.clone(), source.clone()))
            .collect();
            
        for (_script_name, source) in sources {
            let updated_script = DrawingParser::parse_script_with_time(&source, time);
            self.update_script(updated_script);
        }
    }
    
    pub fn update_script(&mut self, script: DrawingScript) {
        if let Some(existing) = self.scripts.iter_mut().find(|s| s.name == script.name) {
            existing.commands = script.commands;
        } else {
            self.scripts.push(script);
        }
    }
    
    pub fn assign_script_to_area(&mut self, script_name: String, area_name: String) {
        // Remove any existing assignment for this script
        self.assignments.retain(|a| a.script_name != script_name);
        
        // Add new assignment
        self.assignments.push(ScriptAssignment {
            script_name,
            area_name,
        });
    }
    
    pub fn get_assigned_shapes(&self, interpreter: &DrawingInterpreter) -> Vec<DrawingShape> {
        let mut shapes = Vec::new();
        
        for assignment in &self.assignments {
            if let Some(script) = self.scripts.iter().find(|s| s.name == assignment.script_name) {
                if let Some(area) = self.areas.iter().find(|a| a.name == assignment.area_name) {
                    let script_shapes = interpreter.create_shapes_in_area(&script.commands, area);
                    shapes.extend(script_shapes);
                }
            }
        }
        
        shapes
    }
}

#[derive(Debug, Clone)]
pub enum DrawCommand {
    Circle { x: f32, y: f32, radius: f32, color: Color },
    Rectangle { x: f32, y: f32, width: f32, height: f32, color: Color },
    Line { x1: f32, y1: f32, x2: f32, y2: f32, color: Color },
    FilledCircle { x: f32, y: f32, radius: f32, color: Color },
    FilledRectangle { x: f32, y: f32, width: f32, height: f32, color: Color, rotation: f32 },
    Polygon { x: f32, y: f32, radius: f32, sides: u32, color: Color, rotation: f32 },
    Clear,
}

#[derive(Debug, Clone)]
pub enum Color {
    Red,
    Blue,
    Green,
    Yellow,
    White,
    Black,
    Hex(String),
    Rgb(f32, f32, f32),
}

impl Color {
    fn to_srgba(&self) -> Srgba {
        match self {
            Color::Red => Srgba::RED,
            Color::Blue => Srgba::BLUE,
            Color::Green => Srgba::GREEN,
            Color::Yellow => Srgba::new(255, 255, 0, 255),
            Color::White => Srgba::WHITE,
            Color::Black => Srgba::BLACK,
            Color::Hex(hex) => {
                // Simple hex parser for #RRGGBB format
                if hex.len() == 7 && hex.starts_with('#') {
                    let r = u8::from_str_radix(&hex[1..3], 16).unwrap_or(0);
                    let g = u8::from_str_radix(&hex[3..5], 16).unwrap_or(0);
                    let b = u8::from_str_radix(&hex[5..7], 16).unwrap_or(0);
                    Srgba::new(r, g, b, 255)
                } else {
                    Srgba::WHITE
                }
            }
            Color::Rgb(r, g, b) => {
                // Clamp values to 0-1 range and convert to 0-255
                let r_clamped = (r.clamp(0.0, 1.0) * 255.0) as u8;
                let g_clamped = (g.clamp(0.0, 1.0) * 255.0) as u8;
                let b_clamped = (b.clamp(0.0, 1.0) * 255.0) as u8;
                Srgba::new(r_clamped, g_clamped, b_clamped, 255)
            }
        }
    }
}

pub enum DrawingShape {
    Circle(Gm<Circle, ColorMaterial>),
    Rectangle(Gm<Rectangle, ColorMaterial>),
    Line(Gm<Line, ColorMaterial>),
    FilledCircle(Gm<Circle, ColorMaterial>),
    FilledRectangle(Gm<Rectangle, ColorMaterial>),
    Polygon(Gm<Circle, ColorMaterial>), // Using circle as base for now
}

pub struct DrawingInterpreter {
    context: Context,
    scale_factor: f32,
    viewport_width: f32,
    viewport_height: f32,
}

impl DrawingInterpreter {
    pub fn new(context: Context, scale_factor: f32, viewport_width: f32, viewport_height: f32) -> Self {
        Self { context, scale_factor, viewport_width, viewport_height }
    }

    pub fn create_shapes_in_area(&self, commands: &[DrawCommand], area: &DrawingArea) -> Vec<DrawingShape> {
        commands
            .iter()
            .filter_map(|cmd| self.command_to_shape_relative(cmd, area))
            .collect()
    }

    fn wrap_coordinate_in_area(&self, coord: f32, area_start: f32, area_size: f32) -> f32 {
        // Coordinates are now relative to the area, so we wrap within 0..area_size
        // then add the area's world position
        let wrapped_relative = coord % area_size;
        let wrapped_relative = if wrapped_relative < 0.0 { wrapped_relative + area_size } else { wrapped_relative };
        area_start + wrapped_relative
    }

    fn command_to_shape_relative(&self, command: &DrawCommand, area: &DrawingArea) -> Option<DrawingShape> {
        match command {
            DrawCommand::Circle { x, y, radius, color } => {
                // Wrap coordinates within area bounds (relative coordinate system)
                let wrapped_x = x % area.width;
                let wrapped_y = y % area.height;
                
                // Translate to area's world position
                let world_x = area.x + wrapped_x;
                let world_y = area.y + wrapped_y;
                
                let circle = Circle::new(
                    &self.context,
                    vec2(world_x, world_y),
                    *radius,
                );
                let material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                Some(DrawingShape::Circle(Gm::new(circle, material)))
            }
            DrawCommand::FilledCircle { x, y, radius, color } => {
                let wrapped_x = x % area.width;
                let wrapped_y = y % area.height;
                let world_x = area.x + wrapped_x;
                let world_y = area.y + wrapped_y;
                
                let circle = Circle::new(
                    &self.context,
                    vec2(world_x, world_y),
                    *radius,
                );
                let mut material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                material.render_states.cull = Cull::None;
                Some(DrawingShape::FilledCircle(Gm::new(circle, material)))
            }
            DrawCommand::FilledRectangle { x, y, width, height, color, rotation } => {
                // Simple coordinate system: just add area offset to input coordinates
                let world_x = area.x + x;
                let world_y = area.y + y;
                
                // Rectangle::new expects center position
                let center_x = world_x + width / 2.0;
                let center_y = world_y + height / 2.0;
                
                
                let rectangle = Rectangle::new(
                    &self.context,
                    vec2(center_x, center_y),
                    degrees(*rotation),
                    *width,
                    *height,
                );
                let mut material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                material.render_states.cull = Cull::None;
                Some(DrawingShape::FilledRectangle(Gm::new(rectangle, material)))
            }
            // Add other shapes as needed...
            _ => None,
        }
    }

    // Keep the original methods for backward compatibility
    fn wrap_coordinate(&self, coord: f32, max_size: f32) -> f32 {
        coord % max_size
    }

    pub fn create_shapes(&self, commands: &[DrawCommand]) -> Vec<DrawingShape> {
        commands
            .iter()
            .filter_map(|cmd| self.command_to_shape(cmd))
            .collect()
    }

    fn command_to_shape(&self, command: &DrawCommand) -> Option<DrawingShape> {
        match command {
            DrawCommand::Circle { x, y, radius, color } => {
                let wrapped_x = self.wrap_coordinate(*x, self.viewport_width);
                let wrapped_y = self.wrap_coordinate(*y, self.viewport_height);
                let circle = Circle::new(
                    &self.context,
                    vec2(wrapped_x * self.scale_factor, wrapped_y * self.scale_factor),
                    *radius * self.scale_factor,
                );
                let material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                Some(DrawingShape::Circle(Gm::new(circle, material)))
            }
            DrawCommand::FilledCircle { x, y, radius, color } => {
                let wrapped_x = self.wrap_coordinate(*x, self.viewport_width);
                let wrapped_y = self.wrap_coordinate(*y, self.viewport_height);
                let circle = Circle::new(
                    &self.context,
                    vec2(wrapped_x * self.scale_factor, wrapped_y * self.scale_factor),
                    *radius * self.scale_factor,
                );
                let mut material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                material.render_states.cull = Cull::None;
                Some(DrawingShape::FilledCircle(Gm::new(circle, material)))
            }
            DrawCommand::Rectangle { x, y, width, height, color } => {
                let wrapped_x = self.wrap_coordinate(*x, self.viewport_width);
                let wrapped_y = self.wrap_coordinate(*y, self.viewport_height);
                let center = vec2(
                    (wrapped_x + width / 2.0) * self.scale_factor,
                    (wrapped_y + height / 2.0) * self.scale_factor,
                );
                let rectangle = Rectangle::new(
                    &self.context,
                    center,
                    degrees(0.0),
                    *width * self.scale_factor,
                    *height * self.scale_factor,
                );
                let material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                Some(DrawingShape::Rectangle(Gm::new(rectangle, material)))
            }
            DrawCommand::FilledRectangle { x, y, width, height, color, rotation } => {
                let wrapped_x = self.wrap_coordinate(*x, self.viewport_width);
                let wrapped_y = self.wrap_coordinate(*y, self.viewport_height);
                let center = vec2(
                    (wrapped_x + width / 2.0) * self.scale_factor,
                    (wrapped_y + height / 2.0) * self.scale_factor,
                );
                let rectangle = Rectangle::new(
                    &self.context,
                    center,
                    degrees(*rotation),
                    *width * self.scale_factor,
                    *height * self.scale_factor,
                );
                let mut material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                material.render_states.cull = Cull::None;
                Some(DrawingShape::FilledRectangle(Gm::new(rectangle, material)))
            }
            DrawCommand::Polygon { x, y, radius, sides: _sides, color, rotation: _rotation } => {
                let wrapped_x = self.wrap_coordinate(*x, self.viewport_width);
                let wrapped_y = self.wrap_coordinate(*y, self.viewport_height);
                // For now, render as filled circle (proper polygon would need mesh generation)
                let circle = Circle::new(
                    &self.context,
                    vec2(wrapped_x * self.scale_factor, wrapped_y * self.scale_factor),
                    *radius * self.scale_factor,
                );
                let mut material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                material.render_states.cull = Cull::None;
                Some(DrawingShape::Polygon(Gm::new(circle, material)))
            }
            DrawCommand::Line { x1, y1, x2, y2, color } => {
                let wrapped_x1 = self.wrap_coordinate(*x1, self.viewport_width);
                let wrapped_y1 = self.wrap_coordinate(*y1, self.viewport_height);
                let wrapped_x2 = self.wrap_coordinate(*x2, self.viewport_width);
                let wrapped_y2 = self.wrap_coordinate(*y2, self.viewport_height);
                let line = Line::new(
                    &self.context,
                    vec2(wrapped_x1 * self.scale_factor, wrapped_y1 * self.scale_factor),
                    vec2(wrapped_x2 * self.scale_factor, wrapped_y2 * self.scale_factor),
                    2.0 * self.scale_factor,
                );
                let material = ColorMaterial {
                    color: color.to_srgba(),
                    ..Default::default()
                };
                Some(DrawingShape::Line(Gm::new(line, material)))
            }
            DrawCommand::Clear => None,
        }
    }
}

pub struct DrawingParser;

impl DrawingParser {
    pub fn parse(input: &str) -> Vec<DrawCommand> {
        input
            .lines()
            .filter_map(|line| Self::parse_line(line.trim()))
            .collect()
    }

    pub fn parse_with_time(input: &str, time: f32) -> Vec<DrawCommand> {
        let time_substituted = Self::substitute_time_variables(input, time);
        Self::parse(&time_substituted)
    }

    pub fn parse_script(input: &str) -> DrawingScript {
        let lines: Vec<&str> = input.lines().collect();
        
        // Look for script name in first line
        let mut script_name = "untitled".to_string();
        let mut start_line = 0;
        
        if let Some(first_line) = lines.first() {
            let trimmed = first_line.trim();
            if trimmed.starts_with("@") {
                script_name = trimmed[1..].trim().to_string();
                start_line = 1;
            }
        }
        
        // Parse remaining lines as commands
        let commands: Vec<DrawCommand> = lines[start_line..]
            .iter()
            .filter_map(|line| Self::parse_line(line.trim()))
            .collect();
            
        DrawingScript {
            name: script_name,
            commands,
        }
    }

    pub fn parse_script_with_time(input: &str, time: f32) -> DrawingScript {
        let time_substituted = Self::substitute_time_variables(input, time);
        Self::parse_script(&time_substituted)
    }

    pub fn substitute_time_variables(input: &str, time: f32) -> String {
        use regex::Regex;
        
        let mut result = input.to_string();
        
        // First, handle sin() and cos() functions before substituting time
        let sin_regex = Regex::new(r"sin\(([^)]+)\)").unwrap();
        result = sin_regex.replace_all(&result, |caps: &regex::Captures| {
            let expr = &caps[1];
            if let Some(val) = Self::evaluate_simple_expression(expr, time) {
                val.sin().to_string()
            } else {
                caps[0].to_string()
            }
        }).to_string();
        
        let cos_regex = Regex::new(r"cos\(([^)]+)\)").unwrap();
        result = cos_regex.replace_all(&result, |caps: &regex::Captures| {
            let expr = &caps[1];
            if let Some(val) = Self::evaluate_simple_expression(expr, time) {
                val.cos().to_string()
            } else {
                caps[0].to_string()
            }
        }).to_string();
        
        // Then substitute time and evaluate any remaining expressions
        result = result.replace("time", &time.to_string());
        
        // Try to evaluate any remaining simple arithmetic expressions (more flexible regex)
        // Run this multiple times to catch nested expressions
        for _ in 0..3 {
            let expr_regex = Regex::new(r"\d+\.?\d*[+\-*/]\d+\.?\d*").unwrap();
            let old_result = result.clone();
            result = expr_regex.replace_all(&result, |caps: &regex::Captures| {
                if let Some(val) = Self::evaluate_simple_expression(&caps[0], time) {
                    val.to_string()
                } else {
                    caps[0].to_string()
                }
            }).to_string();
            
            // Break if no more changes
            if result == old_result {
                break;
            }
        }
        
        result
    }
    
    fn evaluate_simple_expression(expr: &str, time: f32) -> Option<f32> {
        // Replace time with actual value
        let expr = expr.replace("time", &time.to_string());
        
        // Handle simple arithmetic: a*b, a+b, a-b, a/b
        if let Some(pos) = expr.find('*') {
            let (left, right) = expr.split_at(pos);
            let right = &right[1..]; // Skip the '*'
            if let (Ok(a), Ok(b)) = (left.trim().parse::<f32>(), right.trim().parse::<f32>()) {
                return Some(a * b);
            }
        }
        if let Some(pos) = expr.find('+') {
            let (left, right) = expr.split_at(pos);
            let right = &right[1..]; // Skip the '+'
            if let (Ok(a), Ok(b)) = (left.trim().parse::<f32>(), right.trim().parse::<f32>()) {
                return Some(a + b);
            }
        }
        if let Some(pos) = expr.rfind('-') { // rfind to handle negative numbers
            if pos > 0 { // Make sure it's not a leading minus
                let (left, right) = expr.split_at(pos);
                let right = &right[1..]; // Skip the '-'
                if let (Ok(a), Ok(b)) = (left.trim().parse::<f32>(), right.trim().parse::<f32>()) {
                    return Some(a - b);
                }
            }
        }
        if let Some(pos) = expr.find('/') {
            let (left, right) = expr.split_at(pos);
            let right = &right[1..]; // Skip the '/'
            if let (Ok(a), Ok(b)) = (left.trim().parse::<f32>(), right.trim().parse::<f32>()) {
                if b != 0.0 {
                    return Some(a / b);
                }
            }
        }
        
        // If no arithmetic, try to parse as a simple number
        expr.trim().parse::<f32>().ok()
    }

    fn parse_line(line: &str) -> Option<DrawCommand> {
        if line.is_empty() || line.trim().starts_with("//") {
            return None;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        match parts[0] {
            "circle" if parts.len() == 5 => {
                let x = parts[1].parse().ok()?;
                let y = parts[2].parse().ok()?;
                let radius = parts[3].parse().ok()?;
                let color = Self::parse_color(parts[4])?;
                Some(DrawCommand::Circle { x, y, radius, color })
            }
            "filled_circle" if parts.len() == 5 => {
                let x = parts[1].parse().ok()?;
                let y = parts[2].parse().ok()?;
                let radius = parts[3].parse().ok()?;
                let color = Self::parse_color(parts[4])?;
                Some(DrawCommand::FilledCircle { x, y, radius, color })
            }
            "rectangle" if parts.len() == 6 => {
                let x = parts[1].parse().ok()?;
                let y = parts[2].parse().ok()?;
                let width = parts[3].parse().ok()?;
                let height = parts[4].parse().ok()?;
                let color = Self::parse_color(parts[5])?;
                Some(DrawCommand::Rectangle { x, y, width, height, color })
            }
            "filled_rectangle" if parts.len() == 7 => {
                let x = parts[1].parse().ok()?;
                let y = parts[2].parse().ok()?;
                let width = parts[3].parse().ok()?;
                let height = parts[4].parse().ok()?;
                let color = Self::parse_color(parts[5])?;
                let rotation = parts[6].parse().ok()?;
                Some(DrawCommand::FilledRectangle { x, y, width, height, color, rotation })
            }
            "polygon" if parts.len() == 7 => {
                let x = parts[1].parse().ok()?;
                let y = parts[2].parse().ok()?;
                let radius = parts[3].parse().ok()?;
                let sides = parts[4].parse().ok()?;
                let color = Self::parse_color(parts[5])?;
                let rotation = parts[6].parse().ok()?;
                Some(DrawCommand::Polygon { x, y, radius, sides, color, rotation })
            }
            "line" if parts.len() == 6 => {
                let x1 = parts[1].parse().ok()?;
                let y1 = parts[2].parse().ok()?;
                let x2 = parts[3].parse().ok()?;
                let y2 = parts[4].parse().ok()?;
                let color = Self::parse_color(parts[5])?;
                Some(DrawCommand::Line { x1, y1, x2, y2, color })
            }
            "clear" if parts.len() == 1 => Some(DrawCommand::Clear),
            _ => None,
        }
    }

    fn parse_color(color_str: &str) -> Option<Color> {
        // Check for color(r,g,b) format
        if color_str.starts_with("color(") && color_str.ends_with(')') {
            let inner = &color_str[6..color_str.len()-1]; // Remove "color(" and ")"
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() == 3 {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    parts[0].trim().parse::<f32>(),
                    parts[1].trim().parse::<f32>(),
                    parts[2].trim().parse::<f32>()
                ) {
                    return Some(Color::Rgb(r, g, b));
                }
            }
        }
        
        // Fall back to named colors and hex
        match color_str.to_lowercase().as_str() {
            "red" => Some(Color::Red),
            "blue" => Some(Color::Blue),
            "green" => Some(Color::Green),
            "yellow" => Some(Color::Yellow),
            "white" => Some(Color::White),
            "black" => Some(Color::Black),
            hex if hex.starts_with('#') && hex.len() == 7 => Some(Color::Hex(hex.to_string())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_substitution() {
        let input = "filled_rectangle time 100 80 60 color(1,0.5,sin(time*0.2)*0.5+0.5) 0";
        let result = DrawingParser::substitute_time_variables(input, 2.0);
        println!("Input: {}", input);
        println!("Result: {}", result);
        
        // Test simple time substitution
        assert!(result.contains("2"));
    }
    
    #[test]
    fn test_simple_expression_evaluation() {
        assert_eq!(DrawingParser::evaluate_simple_expression("2*0.2", 0.0), Some(0.4));
        assert_eq!(DrawingParser::evaluate_simple_expression("1+0.5", 0.0), Some(1.5));
        assert_eq!(DrawingParser::evaluate_simple_expression("0.5+0.5", 0.0), Some(1.0));
    }
    
    #[test]
    fn test_sin_evaluation() {
        let result = DrawingParser::substitute_time_variables("sin(time*0.2)", 3.14159);
        println!("sin(time*0.2) with time=π: {}", result);
        
        // Should be close to sin(π*0.2) ≈ sin(0.628) ≈ 0.588
        if let Ok(val) = result.parse::<f32>() {
            assert!((val - 0.588).abs() < 0.1);
        }
    }
    
    #[test]
    fn test_color_parsing() {
        let color = DrawingParser::parse_color("color(1,0.5,0.8)");
        assert!(matches!(color, Some(Color::Rgb(1.0, 0.5, 0.8))));
        
        let color2 = DrawingParser::parse_color("red");
        assert!(matches!(color2, Some(Color::Red)));
    }
    
    #[test]
    fn test_full_parsing() {
        let input = "filled_rectangle 100 100 80 60 red 0";
        let commands = DrawingParser::parse(input);
        assert_eq!(commands.len(), 1);
        println!("Parsed: {:?}", commands);
    }
    
    #[test]
    fn test_arithmetic_evaluation() {
        let input = "0.19470917+0.5";
        let result = DrawingParser::evaluate_simple_expression(input, 0.0);
        println!("Evaluating {}: {:?}", input, result);
        assert!(result.is_some());
        assert!((result.unwrap() - 0.69470917).abs() < 0.001);
    }
    
    #[test]
    fn test_regex_matching() {
        let input = "0.19470917+0.5";
        let regex = regex::Regex::new(r"\d+\.?\d*[+\-*/]\d+\.?\d*").unwrap();
        let matches: Vec<_> = regex.find_iter(input).collect();
        println!("Regex matches for '{}': {:?}", input, matches);
        assert!(!matches.is_empty());
    }
    
    #[test]
    fn test_substitution_step_by_step() {
        let input = "color(1,0.5,0.19470917+0.5)";
        println!("Input: {}", input);
        
        let result = DrawingParser::substitute_time_variables(input, 0.0);
        println!("After substitution: {}", result);
        
        let color = DrawingParser::parse_color(&result);
        println!("Parsed color: {:?}", color);
    }

    #[test]
    fn test_complex_substitution() {
        let input = "filled_rectangle time 100 80 60 color(1,0.5,sin(time*0.2)*0.5+0.5) 0";
        let result = DrawingParser::substitute_time_variables(input, 2.0);
        println!("Complex substitution result: {}", result);
        
        // Test if the arithmetic was evaluated
        assert!(!result.contains("+0.5"));
        
        let commands = DrawingParser::parse(&result);
        println!("Parsed commands: {:?}", commands);
        assert_eq!(commands.len(), 1);
    }
    
    #[test]
    fn test_script_parsing() {
        let input = "@moving_rect\nfilled_rectangle time 50 80 60 red 0\ncircle 100 100 20 blue";
        let script = DrawingParser::parse_script(input);
        
        println!("Script: {:?}", script);
        assert_eq!(script.name, "moving_rect");
        assert_eq!(script.commands.len(), 2);
    }
    
    #[test]
    fn test_script_manager() {
        let mut manager = ScriptManager::new();
        
        // Add areas
        let main_area = DrawingArea::new("main".to_string(), 0.0, 0.0, 640.0, 480.0);
        let side_area = DrawingArea::new("side".to_string(), 640.0, 0.0, 640.0, 480.0);
        
        manager.add_area(main_area);
        manager.add_area(side_area);
        
        // Add scripts
        let script1 = "@circle_script\nfilled_circle 100 100 50 red";
        let script2 = "@rect_script\nfilled_rectangle 200 200 80 60 blue 0";
        
        manager.parse_and_add_script(script1, 0.0);
        manager.parse_and_add_script(script2, 0.0);
        
        // Test assignments
        manager.assign_script_to_area("circle_script".to_string(), "main".to_string());
        manager.assign_script_to_area("rect_script".to_string(), "side".to_string());
        
        assert_eq!(manager.scripts.len(), 2);
        assert_eq!(manager.assignments.len(), 2);
        
        // Verify assignment mapping
        let circle_assignment = manager.assignments.iter().find(|a| a.script_name == "circle_script").unwrap();
        assert_eq!(circle_assignment.area_name, "main");
        
        let rect_assignment = manager.assignments.iter().find(|a| a.script_name == "rect_script").unwrap();
        assert_eq!(rect_assignment.area_name, "side");
        
        println!("Script manager test passed!");
    }
    
    #[test]
    fn test_multi_script_simultaneous_execution() {
        let mut manager = ScriptManager::new();
        
        // Set up areas
        let main_area = DrawingArea::new("main".to_string(), 0.0, 0.0, 320.0, 240.0);
        let side_area = DrawingArea::new("side".to_string(), 320.0, 0.0, 320.0, 240.0);
        
        manager.add_area(main_area);
        manager.add_area(side_area);
        
        // Create multiple scripts with time-based animations
        let script1 = "@animated_circle\nfilled_circle sin(time)*50+100 cos(time)*50+100 30 color(1,0,sin(time)*0.5+0.5)";
        let script2 = "@moving_rect\nfilled_rectangle time*10 100 40 30 color(0,1,0) 0";
        let script3 = "@static_shapes\nfilled_circle 200 150 20 red\ncircle 250 180 25 blue";
        
        // Add all scripts
        manager.parse_and_add_script(script1, 0.0);
        manager.parse_and_add_script(script2, 0.0);
        manager.parse_and_add_script(script3, 0.0);
        
        // Assign to different areas
        manager.assign_script_to_area("animated_circle".to_string(), "main".to_string());
        manager.assign_script_to_area("moving_rect".to_string(), "main".to_string());
        manager.assign_script_to_area("static_shapes".to_string(), "side".to_string());
        
        // Test time updates
        manager.update_scripts_with_time(1.0);
        
        // Verify all scripts are present and assigned
        assert_eq!(manager.scripts.len(), 3);
        assert_eq!(manager.assignments.len(), 3);
        
        // Check that scripts assigned to main area have correct assignments
        let main_assignments: Vec<_> = manager.assignments.iter()
            .filter(|a| a.area_name == "main")
            .collect();
        assert_eq!(main_assignments.len(), 2);
        
        let side_assignments: Vec<_> = manager.assignments.iter()
            .filter(|a| a.area_name == "side")
            .collect();
        assert_eq!(side_assignments.len(), 1);
        
        println!("Multi-script simultaneous execution test passed!");
    }
    
    #[test]
    fn test_dynamic_areas_with_relative_coordinates() {
        let mut manager = ScriptManager::new();
        
        // Create a dynamic area
        manager.create_area("test_area".to_string(), 100.0, 50.0, 200.0, 150.0);
        
        // Verify area was created
        assert_eq!(manager.areas.len(), 1);
        let area = &manager.areas[0];
        assert_eq!(area.name, "test_area");
        assert_eq!(area.x, 100.0);
        assert_eq!(area.y, 50.0);
        assert_eq!(area.width, 200.0);
        assert_eq!(area.height, 150.0);
        
        // Test relative coordinate conversion
        let (rel_x, rel_y) = area.to_relative_coords(150.0, 100.0);
        assert_eq!(rel_x, 50.0); // 150 - 100 = 50
        assert_eq!(rel_y, 50.0); // 100 - 50 = 50
        
        // Test world coordinate conversion
        let (world_x, world_y) = area.to_world_coords(50.0, 50.0);
        assert_eq!(world_x, 150.0); // 100 + 50 = 150
        assert_eq!(world_y, 100.0); // 50 + 50 = 100
        
        // Test point containment
        assert!(area.contains_point(150.0, 100.0)); // Inside
        assert!(!area.contains_point(50.0, 25.0)); // Outside
        
        // Test area updates
        manager.update_area("test_area", 200.0, 100.0, 300.0, 200.0);
        let updated_area = &manager.areas[0];
        assert_eq!(updated_area.x, 200.0);
        assert_eq!(updated_area.y, 100.0);
        assert_eq!(updated_area.width, 300.0);
        assert_eq!(updated_area.height, 200.0);
        
        // Test area removal
        manager.remove_area("test_area");
        assert_eq!(manager.areas.len(), 0);
        
        println!("Dynamic areas with relative coordinates test passed!");
    }
}

pub fn main() {
    let window = Window::new(WindowSettings {
        title: "AI Canvas".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();
    let scale_factor = window.device_pixel_ratio();
    
    let mut gui = three_d::GUI::new(&context);
    let text_content = String::from("@moving_rect\nfilled_rectangle 0 0 100 100 color(1,1,0) 0");
    let (initial_width, initial_height) = window.size();
    let interpreter = DrawingInterpreter::new(context.clone(), scale_factor, initial_width as f32, initial_height as f32);
    let drawing_commands: Vec<DrawCommand> = Vec::new();
    let mut auto_update = true;
    
    // Create script manager and set up initial areas
    let mut script_manager = ScriptManager::new();
    
    // Create default areas (account for the 400px left UI panel)
    let ui_panel_width = 400.0;
    let drawable_width = initial_width as f32 - ui_panel_width;
    
    script_manager.create_area("main".to_string(), ui_panel_width, 0.0, drawable_width / 2.0, initial_height as f32);
    script_manager.create_area("side".to_string(), ui_panel_width + drawable_width / 2.0, 0.0, drawable_width / 2.0, initial_height as f32);
    
    
    // Initialize with a default script
    script_manager.parse_and_add_script(&text_content, 0.0);
    script_manager.assign_script_to_area("moving_rect".to_string(), "main".to_string());
    
    
    // Variables for UI
    let mut active_script_tab = 0usize;
    let mut new_script_name = String::new();
    let mut active_area_tab = 0usize;
    let mut new_area_name = String::new();
    let mut new_area_x = 100.0f32;
    let mut new_area_y = 100.0f32;
    let mut new_area_width = 200.0f32;
    let mut new_area_height = 200.0f32;
    let mut show_area_editor = false;
    let mut show_sidebar = true;
    
    window.render_loop(move |mut frame_input| {
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                use three_d::egui;
                
                // Top panel with toggle button
                egui::TopBottomPanel::top("top_panel").show(gui_context, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button(if show_sidebar { "Hide Sidebar" } else { "Show Sidebar" }).clicked() {
                            show_sidebar = !show_sidebar;
                        }
                        ui.label("AI Canvas - Multi-Script Editor");
                    });
                });
                
                if show_sidebar {
                    egui::SidePanel::left("script_panel")
                        .min_width(400.0)
                        .show(gui_context, |ui| {
                        ui.heading("Multi-Script Editor");
                        
                        // Global controls
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut auto_update, "Auto-update with time");
                            if ui.button("Clear All Scripts").clicked() {
                                script_manager.scripts.clear();
                                script_manager.assignments.clear();
                                script_manager.script_sources.clear();
                                active_script_tab = 0;
                            }
                        });
                        
                        ui.separator();
                        
                        // New script creation
                        ui.horizontal(|ui| {
                            ui.label("New Script:");
                            ui.text_edit_singleline(&mut new_script_name);
                            if ui.button("Create Script").clicked() && !new_script_name.is_empty() {
                                let script_content = format!("@{}\n// Add your drawing commands here\nfilled_circle 100 100 30 red", new_script_name);
                                script_manager.parse_and_add_script(&script_content, 0.0);
                                active_script_tab = script_manager.scripts.len().saturating_sub(1);
                                new_script_name.clear();
                            }
                        });
                        
                        ui.separator();
                        
                        // Script tabs
                        if !script_manager.scripts.is_empty() {
                            ui.horizontal(|ui| {
                                for (i, script) in script_manager.scripts.iter().enumerate() {
                                    if ui.selectable_label(active_script_tab == i, &script.name).clicked() {
                                        active_script_tab = i;
                                    }
                                }
                            });
                            
                            ui.separator();
                            
                            // Current script editor
                            if let Some(current_script_name) = script_manager.scripts.get(active_script_tab).map(|s| s.name.clone()) {
                                ui.label(format!("Editing: {}", current_script_name));
                                
                                // Get or create script content
                                let mut script_content = script_manager.script_sources
                                    .get(&current_script_name)
                                    .cloned()
                                    .unwrap_or_else(|| format!("@{}\n// Script content", current_script_name));
                                
                                ui.add(
                                    egui::TextEdit::multiline(&mut script_content)
                                        .min_size([350.0, 150.0].into())
                                        .hint_text("Type your drawing commands here...")
                                );
                                
                                // Update script when content changes
                                if script_manager.script_sources.get(&current_script_name) != Some(&script_content) {
                                    script_manager.script_sources.insert(current_script_name.clone(), script_content.clone());
                                    let time_val = if auto_update { frame_input.accumulated_time as f32 } else { 0.0 };
                                    script_manager.parse_and_add_script(&script_content, time_val);
                                }
                                
                                ui.horizontal(|ui| {
                                    if ui.button("Delete Script").clicked() {
                                        script_manager.scripts.retain(|s| s.name != current_script_name);
                                        script_manager.assignments.retain(|a| a.script_name != current_script_name);
                                        script_manager.script_sources.remove(&current_script_name);
                                        active_script_tab = active_script_tab.saturating_sub(1);
                                    }
                                    
                                    if ui.button("Duplicate Script").clicked() {
                                        let new_name = format!("{}_copy", current_script_name);
                                        let content = script_content.replace(&format!("@{}", current_script_name), &format!("@{}", new_name));
                                        script_manager.parse_and_add_script(&content, 0.0);
                                    }
                                });
                            }
                        } else {
                            ui.label("No scripts created yet. Create your first script above!");
                        }
                        
                        ui.separator();
                        ui.label("Script Assignment:");
                        
                        // Area assignment for all scripts
                        let script_names: Vec<String> = script_manager.scripts.iter().map(|s| s.name.clone()).collect();
                        let area_names: Vec<String> = script_manager.areas.iter().map(|a| a.name.clone()).collect();
                        
                        for script_name in &script_names {
                            ui.horizontal(|ui| {
                                ui.label(format!("{}:", script_name));
                                
                                // Find current assignment
                                let current_area = script_manager.assignments
                                    .iter()
                                    .find(|a| a.script_name == *script_name)
                                    .map(|a| a.area_name.clone())
                                    .unwrap_or_else(|| "none".to_string());
                                
                                let mut selected_area_for_script = current_area.clone();
                                egui::ComboBox::from_id_salt(format!("area_selector_{}", script_name))
                                    .selected_text(&selected_area_for_script)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut selected_area_for_script, "none".to_string(), "none");
                                        for area_name in &area_names {
                                            ui.selectable_value(&mut selected_area_for_script, area_name.clone(), area_name);
                                        }
                                    });
                                
                                // Update assignment if changed
                                if selected_area_for_script != current_area {
                                    if selected_area_for_script == "none" {
                                        script_manager.assignments.retain(|a| a.script_name != *script_name);
                                    } else {
                                        script_manager.assign_script_to_area(script_name.clone(), selected_area_for_script);
                                    }
                                }
                            });
                        }
                        
                        // Show active assignments summary
                        if !script_manager.assignments.is_empty() {
                            ui.separator();
                            ui.label("Active Assignments:");
                            for assignment in &script_manager.assignments {
                                ui.label(format!("• {} → {}", assignment.script_name, assignment.area_name));
                            }
                        }
                        
                        ui.separator();
                        
                        // Area Management Section
                        ui.horizontal(|ui| {
                            ui.label("Area Management:");
                            ui.checkbox(&mut show_area_editor, "Show Area Editor");
                        });
                        
                        if show_area_editor {
                            ui.group(|ui| {
                                ui.label("Create New Area:");
                                ui.horizontal(|ui| {
                                    ui.label("Name:");
                                    ui.text_edit_singleline(&mut new_area_name);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    ui.add(egui::DragValue::new(&mut new_area_x).speed(1.0));
                                    ui.label("Y:");
                                    ui.add(egui::DragValue::new(&mut new_area_y).speed(1.0));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Width:");
                                    ui.add(egui::DragValue::new(&mut new_area_width).speed(1.0).range(50.0..=800.0));
                                    ui.label("Height:");
                                    ui.add(egui::DragValue::new(&mut new_area_height).speed(1.0).range(50.0..=600.0));
                                });
                                
                                if ui.button("Create Area").clicked() && !new_area_name.is_empty() {
                                    script_manager.create_area(new_area_name.clone(), new_area_x, new_area_y, new_area_width, new_area_height);
                                    new_area_name.clear();
                                }
                            });
                            
                            ui.separator();
                            
                            // Existing Areas Editor
                            if !script_manager.areas.is_empty() {
                                ui.label("Edit Existing Areas:");
                                
                                // Area tabs
                                ui.horizontal(|ui| {
                                    for (i, area) in script_manager.areas.iter().enumerate() {
                                        if ui.selectable_label(active_area_tab == i, &area.name).clicked() {
                                            active_area_tab = i;
                                        }
                                    }
                                });
                                
                                // Current area editor
                                if let Some(area) = script_manager.areas.get(active_area_tab).cloned() {
                                    ui.group(|ui| {
                                        ui.label(format!("Editing Area: {}", area.name));
                                        
                                        let mut area_x = area.x;
                                        let mut area_y = area.y;
                                        let mut area_width = area.width;
                                        let mut area_height = area.height;
                                        let mut area_visible = area.visible;
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Position:");
                                            ui.add(egui::DragValue::new(&mut area_x).prefix("X: ").speed(1.0));
                                            ui.add(egui::DragValue::new(&mut area_y).prefix("Y: ").speed(1.0));
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Size:");
                                            ui.add(egui::DragValue::new(&mut area_width).prefix("W: ").speed(1.0).range(50.0..=800.0));
                                            ui.add(egui::DragValue::new(&mut area_height).prefix("H: ").speed(1.0).range(50.0..=600.0));
                                        });
                                        
                                        ui.checkbox(&mut area_visible, "Visible");
                                        
                                        // Update area if values changed
                                        if area_x != area.x || area_y != area.y || area_width != area.width || area_height != area.height {
                                            script_manager.update_area(&area.name, area_x, area_y, area_width, area_height);
                                        }
                                        
                                        if area_visible != area.visible {
                                            if let Some(area_mut) = script_manager.get_area_mut(&area.name) {
                                                area_mut.visible = area_visible;
                                            }
                                        }
                                        
                                        ui.horizontal(|ui| {
                                            if ui.button("Delete Area").clicked() {
                                                script_manager.remove_area(&area.name);
                                                active_area_tab = active_area_tab.saturating_sub(1);
                                            }
                                            
                                            if ui.button("Duplicate Area").clicked() {
                                                let new_name = format!("{}_copy", area.name);
                                                script_manager.create_area(new_name, area_x + 20.0, area_y + 20.0, area_width, area_height);
                                            }
                                        });
                                        
                                        ui.label(format!("Coordinate Info: ({:.0}, {:.0}) - ({:.0}, {:.0})", 
                                                 area_x, area_y, area_x + area_width, area_y + area_height));
                                        ui.label("Note: Drawing coordinates are relative to area (0,0 = top-left of area)");
                                    });
                                }
                            }
                        }
                    });
                }
            },
        );

        // Update areas based on sidebar visibility
        let current_ui_panel_width = if show_sidebar { 400.0 } else { 0.0 };
        let current_drawable_width = frame_input.viewport.width as f32 - current_ui_panel_width;
        let current_height = frame_input.viewport.height as f32;
        
        // Only update areas that exist and ensure valid dimensions
        if let Some(main_area) = script_manager.areas.iter().find(|a| a.name == "main") {
            if main_area.x != current_ui_panel_width || main_area.width != current_drawable_width / 2.0 {
                script_manager.update_area("main", current_ui_panel_width, 0.0, current_drawable_width / 2.0, current_height);
            }
        }
        if let Some(side_area) = script_manager.areas.iter().find(|a| a.name == "side") {
            let side_x = current_ui_panel_width + current_drawable_width / 2.0;
            if side_area.x != side_x || side_area.width != current_drawable_width / 2.0 {
                script_manager.update_area("side", side_x, 0.0, current_drawable_width / 2.0, current_height);
            }
        }

        // Update scripts with current time if auto-update is enabled
        if auto_update {
            let time_val = frame_input.accumulated_time as f32;
            script_manager.update_scripts_with_time(time_val);
        }
        
        // Separate collections for each shape type
        let mut circles = Vec::new();
        let mut rectangles = Vec::new();
        let mut lines = Vec::new();
        let mut filled_circles = Vec::new();
        let mut filled_rectangles = Vec::new();
        let mut polygons = Vec::new();
        
        // Get all shapes from assigned scripts
        let assigned_shapes = script_manager.get_assigned_shapes(&interpreter);
        let has_assigned_shapes = !assigned_shapes.is_empty();
        
        // DEBUG: Print debug info only occasionally

        
        for shape in assigned_shapes {
            match shape {
                DrawingShape::Circle(circle) => circles.push(circle),
                DrawingShape::Rectangle(rect) => rectangles.push(rect),
                DrawingShape::Line(line) => lines.push(line),
                DrawingShape::FilledCircle(circle) => filled_circles.push(circle),
                DrawingShape::FilledRectangle(rect) => filled_rectangles.push(rect),
                DrawingShape::Polygon(poly) => polygons.push(poly),
            }
        }
        
        // Fallback: render unassigned content in first available area for backward compatibility
        if !has_assigned_shapes && !drawing_commands.is_empty() {
            if let Some(first_area) = script_manager.areas.first() {
                let fallback_shapes = interpreter.create_shapes_in_area(&drawing_commands, first_area);
                for shape in fallback_shapes {
                    match shape {
                        DrawingShape::Circle(circle) => circles.push(circle),
                        DrawingShape::Rectangle(rect) => rectangles.push(rect),
                        DrawingShape::Line(line) => lines.push(line),
                        DrawingShape::FilledCircle(circle) => filled_circles.push(circle),
                        DrawingShape::FilledRectangle(rect) => filled_rectangles.push(rect),
                        DrawingShape::Polygon(poly) => polygons.push(poly),
                    }
                }
            }
        }
        
        // Create area boundary visualizations
        let mut area_boundaries = Vec::new();
        for area in &script_manager.areas {
            if area.visible {
                // Create boundary rectangle (using same coordinate system as shapes)
                let boundary_center = vec2(
                    area.x + area.width / 2.0,
                    area.y + area.height / 2.0,
                );
                
                let boundary = Gm::new(
                    Rectangle::new(
                        &context,
                        boundary_center,
                        degrees(0.0),
                        area.width,
                        area.height,
                    ),
                    ColorMaterial {
                        color: Srgba::new(
                            (area.color[0] * 255.0) as u8,
                            (area.color[1] * 255.0) as u8,
                            (area.color[2] * 255.0) as u8,
                            (area.color[3] * 255.0) as u8,
                        ),
                        ..Default::default()
                    },
                );
                area_boundaries.push(boundary);
            }
        }
        
        // Add the original blue square as a reference
        let (width, height) = (frame_input.viewport.width as f32, frame_input.viewport.height as f32);
        let square_size = 60.0 * scale_factor;
        let margin = 20.0 * scale_factor;
        let square_center = vec2(
            width - margin - square_size / 2.0,
            margin + square_size / 2.0
        );
        
        
        let square = Gm::new(
            Rectangle::new(
                &context,
                square_center,
                degrees(0.0),
                square_size,
                square_size,
            ),
            ColorMaterial {
                color: Srgba::new(0, 255, 0, 255),
                ..Default::default()
            },
        );

        let screen = frame_input.screen();
        screen
            .clear(ClearState::color_and_depth(0.15, 0.15, 0.2, 1.0, 1.0))
            .render(Camera::new_2d(frame_input.viewport), &area_boundaries, &[])
            .render(Camera::new_2d(frame_input.viewport), [&square], &[])
            .render(Camera::new_2d(frame_input.viewport), &filled_rectangles, &[])
            .render(Camera::new_2d(frame_input.viewport), &filled_circles, &[])
            .render(Camera::new_2d(frame_input.viewport), &rectangles, &[])
            .render(Camera::new_2d(frame_input.viewport), &circles, &[])
            .render(Camera::new_2d(frame_input.viewport), &lines, &[])
            .render(Camera::new_2d(frame_input.viewport), &polygons, &[])
            .write(|| gui.render())
            .unwrap();

        FrameOutput::default()
    });
}
