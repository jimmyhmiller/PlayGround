//! Root compatibility facade for the extracted filesystem graph driver.

pub use crate::composition::{
    discover, discover_direct, discover_direct_with_config,
    discover_direct_with_config_and_providers, discover_direct_with_config_providers_and_compiler,
    discover_next_with_config, discover_tanstack_with_config, discover_web_with_config,
    discover_with_all_policies, discover_with_policies,
};
pub use diffpack_default_loader::driver::*;
