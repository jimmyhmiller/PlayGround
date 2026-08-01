//! Integration extension points around framework-neutral loader dispatch.

use std::path::Path;
use std::sync::Arc;

use diffpack_core::ResourceId;
use diffpack_core::transform::{Target, TransformResult};

use crate::module::SpecialModule;

pub type SyntheticCompiler<'a> = dyn FnMut(&Path, &str) -> TransformResult + 'a;

/// Policy for synthesized modules owned by framework integrations.
pub trait SpecialModulePolicy: Send + Sync {
    fn query_module(
        &self,
        _resource: &ResourceId,
        _target: Target,
        _compile: &mut SyntheticCompiler<'_>,
    ) -> Result<Option<SpecialModule>, String> {
        Ok(None)
    }

    fn asset_module(
        &self,
        _path: &Path,
        _bytes: &[u8],
        _public_name: &str,
        _base: &str,
        _responsive_variants: bool,
        _compile: &mut SyntheticCompiler<'_>,
    ) -> Result<Option<SpecialModule>, String> {
        Ok(None)
    }

    fn finalize_module(
        &self,
        _id: &str,
        _target: Target,
        _hmr: bool,
        _jsx: diffpack_core::parser::JsxExtensions,
        _module: &mut SpecialModule,
    ) {
    }
}

#[derive(Debug, Default)]
pub struct NoSpecialModulePolicy;

impl SpecialModulePolicy for NoSpecialModulePolicy {}

/// Ordered composition of special-module policies supplied by integrations.
#[derive(Default)]
pub struct SpecialModulePolicyChain {
    policies: Vec<Arc<dyn SpecialModulePolicy>>,
}

impl SpecialModulePolicyChain {
    pub fn new(policies: Vec<Arc<dyn SpecialModulePolicy>>) -> Self {
        Self { policies }
    }
}

impl SpecialModulePolicy for SpecialModulePolicyChain {
    fn query_module(
        &self,
        resource: &ResourceId,
        target: Target,
        compile: &mut SyntheticCompiler<'_>,
    ) -> Result<Option<SpecialModule>, String> {
        for policy in &self.policies {
            if let Some(module) = policy.query_module(resource, target, compile)? {
                return Ok(Some(module));
            }
        }
        Ok(None)
    }

    fn asset_module(
        &self,
        path: &Path,
        bytes: &[u8],
        public_name: &str,
        base: &str,
        responsive_variants: bool,
        compile: &mut SyntheticCompiler<'_>,
    ) -> Result<Option<SpecialModule>, String> {
        for policy in &self.policies {
            if let Some(module) =
                policy.asset_module(path, bytes, public_name, base, responsive_variants, compile)?
            {
                return Ok(Some(module));
            }
        }
        Ok(None)
    }

    fn finalize_module(
        &self,
        id: &str,
        target: Target,
        hmr: bool,
        jsx: diffpack_core::parser::JsxExtensions,
        module: &mut SpecialModule,
    ) {
        for policy in &self.policies {
            policy.finalize_module(id, target, hmr, jsx, module);
        }
    }
}
