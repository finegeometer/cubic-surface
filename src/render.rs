use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

type GL = web_sys::WebGl2RenderingContext;

pub struct Renderer {
    canvas: web_sys::HtmlCanvasElement,
    gl: GL,
    program: web_sys::WebGlProgram,

    vao: web_sys::WebGlVertexArrayObject,
    vertex_buffer: web_sys::WebGlBuffer,

    uniform_position: web_sys::WebGlUniformLocation,
    uniform_orientation: web_sys::WebGlUniformLocation,
    uniform_coeffs: web_sys::WebGlUniformLocation,
    //
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.gl.delete_program(Some(&self.program));
        self.gl.delete_vertex_array(Some(&self.vao));
        self.gl.delete_buffer(Some(&self.vertex_buffer));
    }
}

impl Renderer {
    pub fn new(canvas: &web_sys::HtmlCanvasElement, fragment_shader_source: &str) -> Self {
        let gl = canvas
            .get_context("webgl2")
            .unwrap_throw()
            .unwrap_throw()
            .dyn_into::<GL>()
            .unwrap_throw();

        let vertex_shader = gl.create_shader(GL::VERTEX_SHADER).unwrap_throw();
        gl.shader_source(&vertex_shader, VERTEX_SHADER_SOURCE);
        gl.compile_shader(&vertex_shader);

        let fragment_shader = gl.create_shader(GL::FRAGMENT_SHADER).unwrap_throw();
        gl.shader_source(&fragment_shader, fragment_shader_source);
        gl.compile_shader(&fragment_shader);

        let program = gl.create_program().unwrap_throw();
        gl.attach_shader(&program, &vertex_shader);
        gl.attach_shader(&program, &fragment_shader);
        gl.link_program(&program);

        gl.delete_shader(Some(&vertex_shader));
        gl.delete_shader(Some(&fragment_shader));

        let vao = gl.create_vertex_array().unwrap_throw();
        gl.bind_vertex_array(Some(&vao));

        let vertex_buffer = gl.create_buffer().unwrap_throw();

        let attribute_pos = gl.get_attrib_location(&program, "pos") as u32;

        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vertex_buffer));
        gl.enable_vertex_attrib_array(attribute_pos);
        gl.vertex_attrib_pointer_with_i32(attribute_pos, 2, GL::FLOAT, false, 2 * 4, 0);

        gl.buffer_data_with_array_buffer_view(
            GL::ARRAY_BUFFER,
            &as_f32_array(&[
                -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
            ])
            .into(),
            GL::STATIC_DRAW,
        );

        Self {
            uniform_position: gl.get_uniform_location(&program, "position").unwrap_throw(),
            uniform_orientation: gl
                .get_uniform_location(&program, "orientation")
                .unwrap_throw(),
            uniform_coeffs: gl.get_uniform_location(&program, "coeffs").unwrap_throw(),

            program,
            vao,
            vertex_buffer,

            gl,
            canvas: canvas.clone(),
        }
    }

    pub fn render(&self, isometry: nalgebra::Isometry3<f32>, coeffs: &[f32]) {
        self.gl.use_program(Some(&self.program));
        self.gl.bind_vertex_array(Some(&self.vao));

        self.gl.uniform_matrix3fv_with_f32_array(
            Some(self.uniform_orientation.as_ref()),
            false,
            &isometry.rotation.to_rotation_matrix().matrix().as_slice(),
        );

        self.gl.uniform3f(
            Some(self.uniform_position.as_ref()),
            isometry.translation.vector[0],
            isometry.translation.vector[1],
            isometry.translation.vector[2],
        );

        self.gl
            .uniform1fv_with_f32_array(Some(self.uniform_coeffs.as_ref()), &coeffs);

        self.gl.viewport(
            0,
            0,
            self.canvas.scroll_width(),
            self.canvas.scroll_height(),
        );
        self.gl.draw_arrays(GL::TRIANGLES, 0, 6);
    }
}

const VERTEX_SHADER_SOURCE: &str = r"#version 300 es

in vec2 pos;
out vec2 xy;

void main() {
    xy = vec2(-pos.x, pos.y);
    gl_Position = vec4(pos, 0.0, 1.0);
}


";

fn as_f32_array(v: &[f32]) -> js_sys::Float32Array {
    let memory_buffer = wasm_bindgen::memory()
        .dyn_into::<js_sys::WebAssembly::Memory>()
        .unwrap_throw()
        .buffer();

    let location = v.as_ptr() as u32 / 4;

    js_sys::Float32Array::new(&memory_buffer).subarray(location, location + v.len() as u32)
}
