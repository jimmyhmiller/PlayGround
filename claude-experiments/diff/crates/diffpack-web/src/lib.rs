//! Framework-neutral HTML and browser integration.

pub mod bundle_benchmark;
pub mod compiler;
pub mod config;
pub mod dev_build;
pub mod dev_control;
pub mod dev_proxy;
pub mod dev_response;
pub mod hmr;
pub mod html_entry;
pub mod http;
pub mod node_proxy;
pub mod policies;
pub mod preview;
pub mod response;
pub mod runtime;
pub mod spa_dev;
pub mod spa_server;
pub mod static_files;
pub mod visualizer;
pub mod watch;
pub mod websocket;

pub use diffpack_default_loader::FilesystemProvider;
