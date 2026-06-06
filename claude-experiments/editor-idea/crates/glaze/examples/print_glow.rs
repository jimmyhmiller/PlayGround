use glaze::{parse, Layer};
use std::collections::HashMap;
fn main() {
    let prog = parse(r#"
        token gold.500 = oklch(0.74 0.12 85)
        token accent.solid = gold.500
        style button(intent) {
          fill accent.solid
          overlay shader {
            let pulse = 0.5 + 0.5*sin(time*2)
            emit smoothstep(0,1,hover) * pulse * vec4(1,1,1,0.25)
          }
        }
    "#).unwrap();
    let v: HashMap<String,String> = [("intent","primary")].iter().map(|(k,x)|(k.to_string(),x.to_string())).collect();
    let c = prog.resolve("button", &v, &[]).unwrap();
    for l in &c.layers {
        if let Layer::Shader(s) = l {
            println!("// used uniforms: {:?}", s.used);
            println!("fn fragment() -> @location(0) vec4<f32> {{");
            print!("{}", s.wgsl_body);
            println!("}}");
        } else { println!("// static layer: {:?}", l); }
    }
}
