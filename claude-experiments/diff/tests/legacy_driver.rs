//! Transitional white-box coverage for the extracted filesystem driver.

#![allow(dead_code, unused_imports)]

use diffpack::composition::*;
use diffpack_core::compiler::transform_module;
use diffpack_core::source_map::{line_count, partition_point_from_hint};
use diffpack_default_loader::asset::{asset_public_name, base64_encode, generate_blur_data_url};

include!("../crates/diffpack-default-loader/src/driver.rs");
include!("support/legacy_driver_tests.rs");
