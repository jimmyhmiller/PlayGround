//! Browser runtime fragments used by rendered bundles.

use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use diffpack_default_loader::resolver_policy::NODE_BUILTINS;

/// Generates the browser `requireNative` binding. Unknown optional packages
/// throw immediately; dynamically named Node built-ins yield a throw-on-use
/// proxy so feature detection can inspect their shape safely.
pub fn require_native() -> String {
    let builtins = NODE_BUILTINS
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"const __nodeBuiltins=new Set([{builtins}]);const requireNative=specifier=>{{const builtin=specifier.startsWith("node:")?specifier.length>5:__nodeBuiltins.has(specifier.split("/")[0]);if(!builtin)throw new Error("Cannot require "+JSON.stringify(specifier)+" in the browser: it is not a Node built-in and was not included in the bundle (its specifier is only known at runtime)");const fail=()=>{{throw new Error("node builtin "+specifier+" is not available in the browser");}};const absent=p=>p==="then"||p===Symbol.toPrimitive||p===Symbol.iterator||p===Symbol.asyncIterator;const stub=new Proxy(function(){{fail();}},{{get:(_,p)=>absent(p)?undefined:stub,getOwnPropertyDescriptor:(target,p)=>Reflect.getOwnPropertyDescriptor(target,p)??(typeof p==="string"&&!absent(p)?{{value:stub,writable:true,enumerable:false,configurable:true}}:undefined),has:(target,p)=>absent(p)?Reflect.has(target,p):true,construct:()=>stub,apply:()=>fail()}});return stub;}};"#
    )
}

pub fn free_port() -> Result<u16, String> {
    let listener = TcpListener::bind("0.0.0.0:0")
        .map_err(|error| format!("cannot reserve a port for the node runtime: {error}"))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| format!("cannot read reserved port: {error}"))
}

pub fn spawn_node(index_mjs: &Path, port: u16, control_port: u16) -> Result<Child, String> {
    Command::new("node")
        .arg(index_mjs)
        .env("PORT", port.to_string())
        .env("HOST", "127.0.0.1")
        .env("DIFFPACK_HMR_CONTROL_PORT", control_port.to_string())
        .spawn()
        .map_err(|error| {
            format!(
                "cannot start node SSR runtime ({}): {error}",
                index_mjs.display()
            )
        })
}

pub fn restart_node(
    node: &mut Child,
    index_mjs: &Path,
    port: u16,
    control_port: u16,
) -> Result<(), String> {
    let _ = node.kill();
    let _ = node.wait();
    *node = spawn_node(index_mjs, port, control_port)?;
    wait_for_node(port)
}

pub fn wait_for_node(port: u16) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Err(format!(
        "node SSR runtime did not listen on 127.0.0.1:{port} within 15s"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_stub_distinguishes_builtins_from_unknown_packages() {
        let source = require_native();
        assert!(source.contains("\"module\""));
        assert!(source.contains("it is not a Node built-in"));
        assert!(source.contains("throw-on-use") || source.contains("const fail"));
    }
}
