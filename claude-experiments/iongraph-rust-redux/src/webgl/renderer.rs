//! WebGL 2 renderer
//!
//! Handles WebGL context setup, shader compilation, and rendering.

use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlShader,
    WebGlUniformLocation, WebGlVertexArrayObject,
};

use super::scene::{Color, Primitive, Scene};
use super::shaders;
use super::state::{InteractionState, Viewport};
use super::tessellation::{create_rect, create_stroked_rect, TessellatedPath};
use super::text::{FontConfig, TextAtlas};

/// Compiled shader program with uniform locations
pub struct ShaderProgram {
    pub program: WebGlProgram,
    pub uniforms: HashMap<String, WebGlUniformLocation>,
}

impl ShaderProgram {
    pub fn get_uniform(&self, name: &str) -> Option<&WebGlUniformLocation> {
        self.uniforms.get(name)
    }
}

/// Main WebGL renderer
pub struct WebGLRenderer {
    /// WebGL context
    gl: WebGl2RenderingContext,

    /// Canvas element
    canvas: HtmlCanvasElement,

    /// Shape shader program
    shape_program: ShaderProgram,

    /// Text shader program
    text_program: ShaderProgram,

    /// Text atlas
    pub text_atlas: TextAtlas,

    /// Vertex buffer for dynamic geometry
    vertex_buffer: WebGlBuffer,

    /// Index buffer for dynamic geometry
    index_buffer: WebGlBuffer,

    /// VAO for shape rendering
    shape_vao: WebGlVertexArrayObject,

    /// VAO for text rendering
    text_vao: WebGlVertexArrayObject,

    /// Current canvas size
    width: u32,
    height: u32,
}

impl WebGLRenderer {
    /// Create a new WebGL renderer attached to a canvas
    pub fn new(canvas_id: &str) -> Result<Self, JsValue> {
        let window = web_sys::window().ok_or("No window")?;
        let document = window.document().ok_or("No document")?;

        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or("Canvas not found")?
            .dyn_into::<HtmlCanvasElement>()?;

        let gl = canvas
            .get_context("webgl2")?
            .ok_or("WebGL 2 not supported")?
            .dyn_into::<WebGl2RenderingContext>()?;

        // Enable blending for transparency
        gl.enable(WebGl2RenderingContext::BLEND);
        gl.blend_func(
            WebGl2RenderingContext::SRC_ALPHA,
            WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA,
        );

        // Compile shader programs
        let shape_program = Self::create_shader_program(
            &gl,
            shaders::SHAPE_VERTEX,
            shaders::SHAPE_FRAGMENT,
            &[shaders::uniforms::TRANSFORM],
        )?;

        let text_program = Self::create_shader_program(
            &gl,
            shaders::TEXT_VERTEX,
            shaders::TEXT_FRAGMENT,
            &[shaders::uniforms::TRANSFORM, shaders::uniforms::TEXTURE],
        )?;

        // Create buffers
        let vertex_buffer = gl.create_buffer().ok_or("Failed to create vertex buffer")?;
        let index_buffer = gl.create_buffer().ok_or("Failed to create index buffer")?;

        // Create VAOs
        let shape_vao = gl.create_vertex_array().ok_or("Failed to create VAO")?;
        let text_vao = gl.create_vertex_array().ok_or("Failed to create text VAO")?;

        // Set up shape VAO
        gl.bind_vertex_array(Some(&shape_vao));
        gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vertex_buffer));
        gl.bind_buffer(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            Some(&index_buffer),
        );

        // Position attribute (location 0): 2 floats
        gl.vertex_attrib_pointer_with_i32(
            0,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            6 * 4, // stride: 2 pos + 4 color = 6 floats
            0,
        );
        gl.enable_vertex_attrib_array(0);

        // Color attribute (location 1): 4 floats
        gl.vertex_attrib_pointer_with_i32(
            1,
            4,
            WebGl2RenderingContext::FLOAT,
            false,
            6 * 4,
            2 * 4, // offset: 2 floats
        );
        gl.enable_vertex_attrib_array(1);

        // Set up text VAO (position, texcoord, color)
        gl.bind_vertex_array(Some(&text_vao));
        gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vertex_buffer));
        gl.bind_buffer(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            Some(&index_buffer),
        );

        // Position (location 0): 2 floats
        gl.vertex_attrib_pointer_with_i32(
            0,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            8 * 4, // stride: 2 pos + 2 uv + 4 color = 8 floats
            0,
        );
        gl.enable_vertex_attrib_array(0);

        // Texcoord (location 1): 2 floats
        gl.vertex_attrib_pointer_with_i32(
            1,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            8 * 4,
            2 * 4,
        );
        gl.enable_vertex_attrib_array(1);

        // Color (location 2): 4 floats
        gl.vertex_attrib_pointer_with_i32(
            2,
            4,
            WebGl2RenderingContext::FLOAT,
            false,
            8 * 4,
            4 * 4,
        );
        gl.enable_vertex_attrib_array(2);

        gl.bind_vertex_array(None);

        // Create text atlas
        let text_atlas = TextAtlas::new(&gl)?;

        let width = canvas.width();
        let height = canvas.height();

        // Set initial viewport
        gl.viewport(0, 0, width as i32, height as i32);

        Ok(Self {
            gl,
            canvas,
            shape_program,
            text_program,
            text_atlas,
            vertex_buffer,
            index_buffer,
            shape_vao,
            text_vao,
            width,
            height,
        })
    }

    /// Compile a shader program
    fn create_shader_program(
        gl: &WebGl2RenderingContext,
        vertex_src: &str,
        fragment_src: &str,
        uniform_names: &[&str],
    ) -> Result<ShaderProgram, JsValue> {
        let vertex_shader = Self::compile_shader(gl, WebGl2RenderingContext::VERTEX_SHADER, vertex_src)?;
        let fragment_shader =
            Self::compile_shader(gl, WebGl2RenderingContext::FRAGMENT_SHADER, fragment_src)?;

        let program = gl.create_program().ok_or("Failed to create program")?;
        gl.attach_shader(&program, &vertex_shader);
        gl.attach_shader(&program, &fragment_shader);
        gl.link_program(&program);

        if !gl
            .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            let log = gl
                .get_program_info_log(&program)
                .unwrap_or_default();
            return Err(JsValue::from_str(&format!("Program link failed: {}", log)));
        }

        // Get uniform locations
        let mut uniforms = HashMap::new();
        for name in uniform_names {
            if let Some(loc) = gl.get_uniform_location(&program, name) {
                uniforms.insert(name.to_string(), loc);
                web_sys::console::log_1(&format!("Found uniform: {}", name).into());
            } else {
                web_sys::console::warn_1(&format!("Uniform not found: {}", name).into());
            }
        }

        Ok(ShaderProgram { program, uniforms })
    }

    /// Compile a single shader
    fn compile_shader(
        gl: &WebGl2RenderingContext,
        shader_type: u32,
        source: &str,
    ) -> Result<WebGlShader, JsValue> {
        let shader = gl
            .create_shader(shader_type)
            .ok_or("Failed to create shader")?;

        gl.shader_source(&shader, source);
        gl.compile_shader(&shader);

        if !gl
            .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            let log = gl.get_shader_info_log(&shader).unwrap_or_default();
            return Err(JsValue::from_str(&format!("Shader compile failed: {}", log)));
        }

        Ok(shader)
    }

    /// Resize the canvas
    pub fn resize(&mut self, width: u32, height: u32) {
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.gl.viewport(0, 0, width as i32, height as i32);
        self.width = width;
        self.height = height;
    }

    /// Get canvas dimensions
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Clear the canvas
    pub fn clear(&self, color: Color) {
        self.gl.clear_color(color[0], color[1], color[2], color[3]);
        self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
    }

    /// Render a scene
    pub fn render(&mut self, scene: &Scene, viewport: &Viewport, interaction: &InteractionState) {
        // Clear with white
        self.clear([1.0, 1.0, 1.0, 1.0]);

        // Upload text atlas if needed
        self.text_atlas.upload(&self.gl);

        // Get transform matrix
        let transform = viewport.get_transform_matrix();

        // Debug logging
        web_sys::console::log_1(&format!(
            "render: canvas {}x{}, viewport screen {}x{}, scale {}, offset ({}, {})",
            self.width, self.height,
            viewport.screen_width, viewport.screen_height,
            viewport.scale, viewport.offset.x, viewport.offset.y
        ).into());
        web_sys::console::log_1(&format!(
            "transform matrix: [{:.4}, {:.4}, {:.4}, {:.4}]",
            transform[0], transform[5], transform[12], transform[13]
        ).into());

        // Batch primitives by type for efficient rendering
        let mut shape_vertices: Vec<f32> = Vec::new();
        let mut shape_indices: Vec<u32> = Vec::new();
        let mut text_vertices: Vec<f32> = Vec::new();
        let mut text_indices: Vec<u32> = Vec::new();

        // Scene content will be rendered with the proper transform matrix

        for call in &scene.draw_calls {
            // Check for highlight
            let highlight_color = match &call.owner {
                super::scene::DrawCallOwner::Block { block_id } => {
                    if interaction.is_selected(block_id) {
                        Some(super::scene::colors::SELECTION_HIGHLIGHT)
                    } else if interaction.is_hovered(block_id) {
                        Some(super::scene::colors::HOVER_HIGHLIGHT)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            match &call.primitive {
                Primitive::Rect {
                    rect,
                    color,
                    corner_radius: _,
                } => {
                    let tess = create_rect(rect.x, rect.y, rect.width, rect.height);
                    self.add_colored_geometry(
                        &mut shape_vertices,
                        &mut shape_indices,
                        &tess,
                        *color,
                    );

                    // Add highlight overlay if selected/hovered
                    if let Some(hl_color) = highlight_color {
                        let hl_tess = create_rect(rect.x, rect.y, rect.width, rect.height);
                        self.add_colored_geometry(
                            &mut shape_vertices,
                            &mut shape_indices,
                            &hl_tess,
                            hl_color,
                        );
                    }
                }
                Primitive::StrokedRect {
                    rect,
                    color,
                    stroke_width,
                } => {
                    let tess =
                        create_stroked_rect(rect.x, rect.y, rect.width, rect.height, *stroke_width);
                    self.add_colored_geometry(
                        &mut shape_vertices,
                        &mut shape_indices,
                        &tess,
                        *color,
                    );
                }
                Primitive::Triangle { points, color } => {
                    let base = (shape_vertices.len() / 6) as u32;
                    for p in points {
                        shape_vertices.extend_from_slice(&[
                            p[0], p[1], color[0], color[1], color[2], color[3],
                        ]);
                    }
                    shape_indices.extend_from_slice(&[base, base + 1, base + 2]);
                }
                Primitive::Path {
                    vertices,
                    indices,
                    color,
                } => {
                    let base = (shape_vertices.len() / 6) as u32;
                    for i in (0..vertices.len()).step_by(2) {
                        shape_vertices.extend_from_slice(&[
                            vertices[i],
                            vertices[i + 1],
                            color[0],
                            color[1],
                            color[2],
                            color[3],
                        ]);
                    }
                    for idx in indices {
                        shape_indices.push(base + idx);
                    }
                }
                Primitive::Line {
                    x1,
                    y1,
                    x2,
                    y2,
                    color,
                    width,
                } => {
                    // Convert line to quad
                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len > 0.0 {
                        let nx = -dy / len * width / 2.0;
                        let ny = dx / len * width / 2.0;

                        let base = (shape_vertices.len() / 6) as u32;
                        shape_vertices.extend_from_slice(&[
                            x1 + nx,
                            y1 + ny,
                            color[0],
                            color[1],
                            color[2],
                            color[3],
                            x1 - nx,
                            y1 - ny,
                            color[0],
                            color[1],
                            color[2],
                            color[3],
                            x2 - nx,
                            y2 - ny,
                            color[0],
                            color[1],
                            color[2],
                            color[3],
                            x2 + nx,
                            y2 + ny,
                            color[0],
                            color[1],
                            color[2],
                            color[3],
                        ]);
                        shape_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                    }
                }
                Primitive::Text {
                    x,
                    y,
                    text,
                    color,
                    font_size,
                    font_weight,
                    align: _,
                } => {
                    let font = FontConfig::new(
                        "monospace",
                        *font_size as u32,
                        *font_weight,
                    );
                    let block = self.text_atlas.get_or_render(text, &font);

                    // Create textured quad
                    let base = (text_vertices.len() / 8) as u32;
                    let px = *x;
                    let py = *y;
                    let pw = block.width;
                    let ph = block.height;
                    let u0 = block.uv[0];
                    let v0 = block.uv[1];
                    let u1 = block.uv[0] + block.uv[2];
                    let v1 = block.uv[1] + block.uv[3];

                    // 4 vertices: position (2), texcoord (2), color (4)
                    text_vertices.extend_from_slice(&[
                        px,
                        py,
                        u0,
                        v0,
                        color[0],
                        color[1],
                        color[2],
                        color[3], // top-left
                        px + pw,
                        py,
                        u1,
                        v0,
                        color[0],
                        color[1],
                        color[2],
                        color[3], // top-right
                        px + pw,
                        py + ph,
                        u1,
                        v1,
                        color[0],
                        color[1],
                        color[2],
                        color[3], // bottom-right
                        px,
                        py + ph,
                        u0,
                        v1,
                        color[0],
                        color[1],
                        color[2],
                        color[3], // bottom-left
                    ]);
                    text_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
        }

        // Draw shapes
        web_sys::console::log_1(&format!(
            "shapes: {} vertices, {} indices; text: {} vertices, {} indices",
            shape_vertices.len(), shape_indices.len(),
            text_vertices.len(), text_indices.len()
        ).into());

        // Log first few vertices
        if shape_vertices.len() >= 12 {
            web_sys::console::log_1(&format!(
                "first 2 shape verts: ({}, {}) rgba({},{},{},{}), ({}, {}) rgba({},{},{},{})",
                shape_vertices[0], shape_vertices[1],
                shape_vertices[2], shape_vertices[3], shape_vertices[4], shape_vertices[5],
                shape_vertices[6], shape_vertices[7],
                shape_vertices[8], shape_vertices[9], shape_vertices[10], shape_vertices[11],
            ).into());
        }

        if !shape_indices.is_empty() {
            self.gl.use_program(Some(&self.shape_program.program));
            self.gl.bind_vertex_array(Some(&self.shape_vao));

            // Set transform uniform
            let debug_identity = false;
            let matrix_to_use = if debug_identity {
                web_sys::console::log_1(&"DEBUG: Using identity matrix".into());
                [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ]
            } else {
                transform
            };

            if let Some(loc) = self.shape_program.get_uniform(shaders::uniforms::TRANSFORM) {
                self.gl.uniform_matrix4fv_with_f32_array(Some(loc), false, &matrix_to_use);
                web_sys::console::log_1(&"Transform uniform set".into());
            } else {
                web_sys::console::error_1(&"Transform uniform NOT found during render!".into());
            }

            // Upload vertex data
            self.gl.bind_buffer(
                WebGl2RenderingContext::ARRAY_BUFFER,
                Some(&self.vertex_buffer),
            );
            unsafe {
                let view = js_sys::Float32Array::view(&shape_vertices);
                self.gl.buffer_data_with_array_buffer_view(
                    WebGl2RenderingContext::ARRAY_BUFFER,
                    &view,
                    WebGl2RenderingContext::DYNAMIC_DRAW,
                );
            }

            // Upload index data
            self.gl.bind_buffer(
                WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                Some(&self.index_buffer),
            );
            unsafe {
                let view = js_sys::Uint32Array::view(&shape_indices);
                self.gl.buffer_data_with_array_buffer_view(
                    WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                    &view,
                    WebGl2RenderingContext::DYNAMIC_DRAW,
                );
            }

            // Draw
            self.gl.draw_elements_with_i32(
                WebGl2RenderingContext::TRIANGLES,
                shape_indices.len() as i32,
                WebGl2RenderingContext::UNSIGNED_INT,
                0,
            );

            // Check for WebGL errors
            let error = self.gl.get_error();
            if error != WebGl2RenderingContext::NO_ERROR {
                web_sys::console::error_1(&format!("WebGL error after shape draw: {}", error).into());
            }
        }

        // Draw text
        if !text_indices.is_empty() {
            self.gl.use_program(Some(&self.text_program.program));
            self.gl.bind_vertex_array(Some(&self.text_vao));

            // Set transform uniform
            if let Some(loc) = self.text_program.get_uniform(shaders::uniforms::TRANSFORM) {
                self.gl.uniform_matrix4fv_with_f32_array(Some(loc), false, &transform);
            }

            // Bind text atlas texture
            self.text_atlas.bind(&self.gl, 0);
            if let Some(loc) = self.text_program.get_uniform(shaders::uniforms::TEXTURE) {
                self.gl.uniform1i(Some(loc), 0);
            }

            // Upload vertex data
            self.gl.bind_buffer(
                WebGl2RenderingContext::ARRAY_BUFFER,
                Some(&self.vertex_buffer),
            );
            unsafe {
                let view = js_sys::Float32Array::view(&text_vertices);
                self.gl.buffer_data_with_array_buffer_view(
                    WebGl2RenderingContext::ARRAY_BUFFER,
                    &view,
                    WebGl2RenderingContext::DYNAMIC_DRAW,
                );
            }

            // Upload index data
            self.gl.bind_buffer(
                WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                Some(&self.index_buffer),
            );
            unsafe {
                let view = js_sys::Uint32Array::view(&text_indices);
                self.gl.buffer_data_with_array_buffer_view(
                    WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
                    &view,
                    WebGl2RenderingContext::DYNAMIC_DRAW,
                );
            }

            // Draw
            self.gl.draw_elements_with_i32(
                WebGl2RenderingContext::TRIANGLES,
                text_indices.len() as i32,
                WebGl2RenderingContext::UNSIGNED_INT,
                0,
            );
        }

        self.gl.bind_vertex_array(None);
    }

    /// Add colored geometry to vertex/index buffers
    fn add_colored_geometry(
        &self,
        vertices: &mut Vec<f32>,
        indices: &mut Vec<u32>,
        tess: &TessellatedPath,
        color: Color,
    ) {
        let base = (vertices.len() / 6) as u32;

        // Add vertices with color
        for i in (0..tess.vertices.len()).step_by(2) {
            vertices.extend_from_slice(&[
                tess.vertices[i],
                tess.vertices[i + 1],
                color[0],
                color[1],
                color[2],
                color[3],
            ]);
        }

        // Add indices
        for idx in &tess.indices {
            indices.push(base + idx);
        }
    }
}
