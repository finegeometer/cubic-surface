#![forbid(unsafe_code)]

mod fps;
mod render;

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
pub fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    web_sys::window()
        .unwrap_throw()
        .request_animation_frame(&State::new().0.borrow().animation_frame_closure)
        .unwrap_throw();
}

pub enum Msg {
    Click,
    MouseMove([i32; 2]),
    KeyDown(String),
    KeyUp(String),
}

#[derive(Clone)]
struct State(Rc<RefCell<Model>>);

struct Model {
    animation_frame_closure: js_sys::Function,
    keys: HashSet<String>,
    fps: Option<fps::FrameCounter>,
    renderer: render::Renderer,

    window: web_sys::Window,
    document: web_sys::Document,
    canvas: web_sys::HtmlCanvasElement,
    info_box: web_sys::HtmlParagraphElement,

    // Camera -> World
    camera: nalgebra::Isometry3<f32>,
    sliders: [web_sys::HtmlInputElement; 20],
}

impl State {
    fn new() -> Self {
        let out = Self(Rc::new(RefCell::new(Model::new())));

        {
            let model: &mut Model = &mut out.0.borrow_mut();

            out.event_listener(&model.canvas, "mousedown", move |_| Msg::Click);
            out.event_listener(&model.canvas, "mousemove", |evt| {
                let evt = evt.dyn_into::<web_sys::MouseEvent>().unwrap_throw();
                Msg::MouseMove([evt.movement_x(), evt.movement_y()])
            });
            out.event_listener(&model.document, "keydown", |evt| {
                let evt = evt.dyn_into::<web_sys::KeyboardEvent>().unwrap_throw();
                Msg::KeyDown(evt.key())
            });
            out.event_listener(&model.document, "keyup", |evt| {
                let evt = evt.dyn_into::<web_sys::KeyboardEvent>().unwrap_throw();
                Msg::KeyUp(evt.key())
            });

            let state = out.clone();
            let closure: Closure<dyn FnMut(f64)> = Closure::wrap(Box::new(move |timestamp| {
                state.frame(timestamp);
            }));
            model.animation_frame_closure =
                closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
            closure.forget();
        }

        out
    }

    fn update(&self, msg: Msg) {
        let model: &mut Model = &mut self.0.borrow_mut();

        match msg {
            Msg::Click => {
                if model.document.pointer_lock_element().is_none() {
                    model.canvas.request_pointer_lock();
                }
            }
            Msg::KeyDown(k) => {
                model.keys.insert(k.to_lowercase());
            }
            Msg::KeyUp(k) => {
                model.keys.remove(&k.to_lowercase());
            }
            Msg::MouseMove([x, y]) => {
                if model.document.pointer_lock_element().is_some() {
                    model.camera *= &nalgebra::UnitQuaternion::new(nalgebra::Vector3::new(
                        y as f32 * 3e-3,
                        -x as f32 * 3e-3,
                        0.,
                    ));
                }
            }
        }
    }

    fn frame(&self, timestamp: f64) {
        let model: &mut Model = &mut self.0.borrow_mut();

        model
            .window
            .request_animation_frame(&model.animation_frame_closure)
            .unwrap_throw();

        if let Some(fps) = &mut model.fps {
            let dt = fps.frame(timestamp);

            let format_slider = |first: &mut bool, monomial: &str, slider_number: usize| {
                let value = model.sliders[slider_number].value_as_number();
                if value == 0.0 {
                    return String::new();
                }

                #[allow(clippy::float_cmp)]
                fn coefficient(value: f64, monomial: &str) -> String {
                    if monomial == "" {
                        format!("{}", value)
                    } else if value == 1.0 {
                        String::new()
                    } else if value == -1.0 {
                        "-".to_string()
                    } else {
                        format!("{}", value)
                    }
                }

                let out = if *first {
                    format!("{}{}", coefficient(value, monomial), monomial)
                } else if value < 0.0 {
                    format!(" - {}{}", coefficient(-value, monomial), monomial)
                } else {
                    format!(" + {}{}", coefficient(value, monomial), monomial)
                };

                *first = false;

                out
            };

            let mut tmp_bool = true;
            model.info_box.set_inner_text(
                &([
                    "x³", "x²y", "xy²", "y³", "x²z", "xyz", "y²z", "xz²", "yz²", "z³", "x²", "xy",
                    "y²", "xz", "yz", "z²", "x", "y", "z", "",
                ]
                .iter()
                .enumerate()
                .map(|(n, monomial)| format_slider(&mut tmp_bool, monomial, n))
                .collect::<String>()
                    + " = 0"),
            );

            model.move_player(dt as f32);
            model.view();
        } else {
            model.fps = Some(<fps::FrameCounter>::new(timestamp));
        }
    }

    fn event_listener(
        &self,
        target: &web_sys::EventTarget,
        event: &str,
        msg: impl Fn(web_sys::Event) -> Msg + 'static,
    ) {
        let state = self.clone();
        let closure: Closure<dyn FnMut(web_sys::Event)> = Closure::wrap(Box::new(move |evt| {
            state.update(msg(evt));
        }));
        target
            .add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())
            .unwrap_throw();
        closure.forget();
    }
}

impl Model {
    fn new() -> Self {
        let window = web_sys::window().unwrap_throw();
        let document = window.document().unwrap_throw();
        let body = document.body().unwrap_throw();

        let canvas = document
            .create_element("canvas")
            .unwrap_throw()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap_throw();
        canvas.set_attribute("width", "800").unwrap_throw();
        canvas.set_attribute("height", "800").unwrap_throw();
        body.append_child(&canvas).unwrap_throw();

        let info_box = document
            .create_element("p")
            .unwrap_throw()
            .dyn_into::<web_sys::HtmlParagraphElement>()
            .unwrap_throw();
        body.append_child(&info_box).unwrap_throw();

        let make_slider = |value: &'static str| {
            let slider = document
                .create_element("input")
                .unwrap_throw()
                .dyn_into::<web_sys::HtmlInputElement>()
                .unwrap_throw();
            slider.set_type("range");
            slider.set_min("-2");
            slider.set_max("2");
            slider.set_value(value);
            body.append_child(&slider).unwrap_throw();
            slider
        };

        Self {
            animation_frame_closure: JsValue::undefined().into(),
            fps: None,
            keys: HashSet::new(),
            renderer: render::Renderer::new(&canvas, FRAGMENT_SHADER_SOURCE),

            camera: nalgebra::Isometry3::identity(),
            sliders: [
                make_slider("0"),  // XXX
                make_slider("0"),  // XXY
                make_slider("0"),  // XYY
                make_slider("0"),  // YYY
                make_slider("0"),  // XXZ
                make_slider("2"),  // XYZ
                make_slider("0"),  // YYZ
                make_slider("0"),  // XZZ
                make_slider("0"),  // YZZ
                make_slider("0"),  // ZZZ
                make_slider("1"),  // XX
                make_slider("0"),  // XY
                make_slider("1"),  // YY
                make_slider("0"),  // XZ
                make_slider("0"),  // YZ
                make_slider("1"),  // ZZ
                make_slider("0"),  // X
                make_slider("0"),  // Y
                make_slider("0"),  // Z
                make_slider("-2"), //
            ],

            window,
            document,
            canvas,
            info_box,
        }
    }

    fn move_player(&mut self, dt: f32) {
        let speed = 2.;

        let mut v = nalgebra::Vector3::zeros();
        if self.keys.contains(" ") {
            v += nalgebra::Vector3::y() * dt * speed;
        }
        if self.keys.contains("shift") {
            v -= nalgebra::Vector3::y() * dt * speed;
        }
        if self.keys.contains("w") {
            v += nalgebra::Vector3::z() * dt * speed;
        }
        if self.keys.contains("s") {
            v -= nalgebra::Vector3::z() * dt * speed;
        }
        if self.keys.contains("a") {
            v += nalgebra::Vector3::x() * dt * speed;
        }
        if self.keys.contains("d") {
            v -= nalgebra::Vector3::x() * dt * speed;
        }
        self.camera *= &nalgebra::Translation3::from(v);
    }

    fn view(&self) {
        self.renderer.render(
            self.camera,
            &self
                .sliders
                .iter()
                .map(|slider| slider.value_as_number() as f32)
                .collect::<Vec<f32>>(),
        );
    }
}

const FRAGMENT_SHADER_SOURCE: &str = r"#version 300 es

precision mediump float;

in vec2 xy;
out vec4 color;

// xxx  xxy  xyy  yyy  xxz  xyz  yyz  xzz  yzz  zzz  xx  xy  yy  xz  yz  zz  x  y  z  1
uniform float coeffs[20];
uniform vec3 position;
uniform mat3 orientation;


float cbrt(float x) {
    return sign(x) * pow(abs(x), 0.33333333333333333);
}

int depressed_cubic(float p, float q, out vec3 roots) {
        if (p == 0.0) {
            roots = vec3(cbrt(-q));
            return 3;
        }

        // Discriminant
        float discriminant = -(4.0 * p * p * p + 27.0 * q * q);

        if (discriminant >= 0.0) {
            // Three real roots

            float radical = sqrt(-p / 3.0);

            float theta = acos(3.0 * q / (2.0 * p * radical)) / 3.0;

            roots = vec3(
                cos(theta),
                cos(theta + 2.09439510239319549),
                cos(theta - 2.09439510239319549)
            );
            roots *= 2.0 * radical;

            return 3;

        } else {
            // discard;
            // One real root
            float inner_radical = sqrt(-discriminant / 108.0);
            roots = vec3(cbrt(-0.5*q + inner_radical) + cbrt(-0.5*q - inner_radical));
            return 1;
        }

}

int calc_roots(float poly[4], out float roots[3]) {
    if (poly[3] != 0.0) {
        // Cubic
        // Thanks, Wikipedia!

        float b = poly[1] / poly[0];
        float c = poly[2] / poly[0];
        float d = poly[3] / poly[0];


        // Calculate depressed cubic: t^3 + pt + q = 0

        float p = (3.0*c - b*b) / 3.0;
        float q = (2.0*b*b*b - 9.0*b*c + 27.0*d) / 27.0;

        vec3 outputs;
        int num_roots = depressed_cubic(p, q, outputs);

        outputs -= vec3(b / 3.0);
        outputs = vec3(1.0) / outputs;

        if (outputs.x < outputs.z) {
            roots[0] = min(outputs.x, outputs.y);
            roots[1] = min(max(outputs.x, outputs.y), outputs.z);
            roots[2] = max(outputs.y, outputs.z);
        } else {
            roots[0] = min(outputs.z, outputs.y);
            roots[1] = min(max(outputs.z, outputs.y), outputs.x);
            roots[2] = max(outputs.y, outputs.x);
        }

        return num_roots;
    } else if (poly[2] != 0.0) {
        // Quadratic
        float discriminant = poly[1] * poly[1] - 4.0 * poly[2] * poly[0];
        if (discriminant < 0.0) {
            return 0;
        } else {
            vec2 outputs = vec2(sqrt(discriminant));
            outputs *= vec2(-1.0, 1.0);
            outputs -= vec2(poly[1]);
            outputs /= 2.0 * poly[2];

            roots[0] = min(outputs.x, outputs.y);
            roots[1] = max(outputs.x, outputs.y);

            return 2;
        }
    } else if (poly[1] != 0.0) {
        // Linear
        roots[0] = -poly[0] / poly[1];
        return 1;
    } else {
        // Constant
        return 0;
    }
}




vec3 gradient(vec3 pos) {
    return vec3(
        3.0 * pos.x * pos.x * coeffs[0] + 2.0 * pos.x * pos.y * coeffs[1] +       pos.y * pos.y * coeffs[2] + 2.0 * pos.x * pos.z * coeffs[4] +       pos.y * pos.z * coeffs[5] +       pos.z * pos.z * coeffs[7] + 2.0 * pos.x * coeffs[10] +       pos.y * coeffs[11] +       pos.z * coeffs[13] + coeffs[16],
              pos.x * pos.x * coeffs[1] + 2.0 * pos.x * pos.y * coeffs[2] + 3.0 * pos.y * pos.y * coeffs[3] +       pos.x * pos.z * coeffs[5] + 2.0 * pos.y * pos.z * coeffs[6] +       pos.z * pos.z * coeffs[8] +       pos.x * coeffs[11] + 2.0 * pos.y * coeffs[12] +       pos.z * coeffs[14] + coeffs[17],
              pos.x * pos.x * coeffs[4] +       pos.x * pos.y * coeffs[5] +       pos.y * pos.y * coeffs[6] + 2.0 * pos.x * pos.z * coeffs[7] + 2.0 * pos.y * pos.z * coeffs[8] + 3.0 * pos.z * pos.z * coeffs[9] +       pos.x * coeffs[13] +       pos.y * coeffs[14] + 2.0 * pos.z * coeffs[15] + coeffs[18]
    );
}

bool trace(vec3 pos, vec3 dir, out float t) {

    vec4 poly = vec4(0.);
    poly += coeffs[ 0] * vec4(pos.x * pos.x * pos.x, dir.x * pos.x * pos.x + pos.x * dir.x * pos.x + pos.x * pos.x * dir.x, dir.x * dir.x * pos.x + dir.x * pos.x * dir.x + pos.x * dir.x * dir.x, dir.x * dir.x * dir.x);
    poly += coeffs[ 1] * vec4(pos.x * pos.x * pos.y, dir.x * pos.x * pos.y + pos.x * dir.x * pos.y + pos.x * pos.x * dir.y, dir.x * dir.x * pos.y + dir.x * pos.x * dir.y + pos.x * dir.x * dir.y, dir.x * dir.x * dir.y);
    poly += coeffs[ 2] * vec4(pos.x * pos.y * pos.y, dir.x * pos.y * pos.y + pos.x * dir.y * pos.y + pos.x * pos.y * dir.y, dir.x * dir.y * pos.y + dir.x * pos.y * dir.y + pos.x * dir.y * dir.y, dir.x * dir.y * dir.y);
    poly += coeffs[ 3] * vec4(pos.y * pos.y * pos.y, dir.y * pos.y * pos.y + pos.y * dir.y * pos.y + pos.y * pos.y * dir.y, dir.y * dir.y * pos.y + dir.y * pos.y * dir.y + pos.y * dir.y * dir.y, dir.y * dir.y * dir.y);
    poly += coeffs[ 4] * vec4(pos.x * pos.x * pos.z, dir.x * pos.x * pos.z + pos.x * dir.x * pos.z + pos.x * pos.x * dir.z, dir.x * dir.x * pos.z + dir.x * pos.x * dir.z + pos.x * dir.x * dir.z, dir.x * dir.x * dir.z);
    poly += coeffs[ 5] * vec4(pos.x * pos.y * pos.z, dir.x * pos.y * pos.z + pos.x * dir.y * pos.z + pos.x * pos.y * dir.z, dir.x * dir.y * pos.z + dir.x * pos.y * dir.z + pos.x * dir.y * dir.z, dir.x * dir.y * dir.z);
    poly += coeffs[ 6] * vec4(pos.y * pos.y * pos.z, dir.y * pos.y * pos.z + pos.y * dir.y * pos.z + pos.y * pos.y * dir.z, dir.y * dir.y * pos.z + dir.y * pos.y * dir.z + pos.y * dir.y * dir.z, dir.y * dir.y * dir.z);
    poly += coeffs[ 7] * vec4(pos.x * pos.z * pos.z, dir.x * pos.z * pos.z + pos.x * dir.z * pos.z + pos.x * pos.z * dir.z, dir.x * dir.z * pos.z + dir.x * pos.z * dir.z + pos.x * dir.z * dir.z, dir.x * dir.z * dir.z);
    poly += coeffs[ 8] * vec4(pos.y * pos.z * pos.z, dir.y * pos.z * pos.z + pos.y * dir.z * pos.z + pos.y * pos.z * dir.z, dir.y * dir.z * pos.z + dir.y * pos.z * dir.z + pos.y * dir.z * dir.z, dir.y * dir.z * dir.z);
    poly += coeffs[ 9] * vec4(pos.z * pos.z * pos.z, dir.z * pos.z * pos.z + pos.z * dir.z * pos.z + pos.z * pos.z * dir.z, dir.z * dir.z * pos.z + dir.z * pos.z * dir.z + pos.z * dir.z * dir.z, dir.z * dir.z * dir.z);
    poly += coeffs[10] * vec4(pos.x * pos.x, dir.x * pos.x + pos.x * dir.x, dir.x * dir.x, 0.0);
    poly += coeffs[11] * vec4(pos.x * pos.y, dir.x * pos.y + pos.x * dir.y, dir.x * dir.y, 0.0);
    poly += coeffs[12] * vec4(pos.y * pos.y, dir.y * pos.y + pos.y * dir.y, dir.y * dir.y, 0.0);
    poly += coeffs[13] * vec4(pos.x * pos.z, dir.x * pos.z + pos.x * dir.z, dir.x * dir.z, 0.0);
    poly += coeffs[14] * vec4(pos.y * pos.z, dir.y * pos.z + pos.y * dir.z, dir.y * dir.z, 0.0);
    poly += coeffs[15] * vec4(pos.z * pos.z, dir.z * pos.z + pos.z * dir.z, dir.z * dir.z, 0.0);
    poly += coeffs[16] * vec4(pos.x, dir.x, 0.0, 0.0);
    poly += coeffs[17] * vec4(pos.y, dir.y, 0.0, 0.0);
    poly += coeffs[18] * vec4(pos.z, dir.z, 0.0, 0.0);
    poly += coeffs[19] * vec4(1.0, 0.0, 0.0, 0.0);

    float[4] polynomial;
    polynomial[0] = poly.x;
    polynomial[1] = poly.y;
    polynomial[2] = poly.z;
    polynomial[3] = poly.w;

    float roots[3];
    int num_roots = calc_roots(polynomial, roots);

    for (int i = 0; i < num_roots; i++) {
        if (roots[i] >= 0.0) {
            t = roots[i];
            return true;
        }
    }

    return false;
}

float integer_bump(float x) {
    return pow(2.0 * abs(mod(x, 1.0) - 0.5), 16.0);
}

void main() {
    vec3 pos = position;
    vec3 dir = orientation * normalize(vec3(xy, 1.0));

    float t;
    if (!trace(pos, dir, t)) {
        discard;
    }

    vec3 point = pos + t * dir;
    vec3 normal = normalize(gradient(point));

    dir = -dir; // Now the direction *to* the eye

    vec3 base_color = vec3(0.0, 0.3, 1.0);
    if (dot(dir, normal) < 0.0) {
        normal = -normal;
        base_color = vec3(0.7, 0.0, 0.0);
    }
    base_color += vec3(max(max(integer_bump(point.x), integer_bump(point.y)), integer_bump(point.z)));
    base_color = min(base_color, vec3(1.0));

    vec3 light_direction = normalize(vec3(1.0, -2.0, -3.0));

    vec3 ambient = 0.2 * base_color;

    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    if (dot(normal, light_direction) > 0.0) {
        diffuse = 0.8 * dot(normal, light_direction) * base_color;
        specular = pow(max(0.0, dot(normal, normalize(light_direction + dir))), 64.0) * vec3(1.0);
    }

    color = vec4(ambient + diffuse + specular, 1.0);
}

";
