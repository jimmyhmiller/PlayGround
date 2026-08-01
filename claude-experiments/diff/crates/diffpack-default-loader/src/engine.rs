//! Small public facade for embedding Diffpack's filesystem build engine.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use diffpack_core::compiler::{CoreModuleCompiler, ModuleCompiler};
use diffpack_core::runtime::{NoRuntimeIntegrationPolicy, RuntimeIntegrationPolicy};
use diffpack_core::{
    BuildMode, BuildUpdate, Environment, ModuleProvider, Platform, ProviderPipeline,
};

use crate::driver::Bundler;
use crate::driver_config::{BuildConfig, DriverPolicies};
use crate::module_policy::{NoSpecialModulePolicy, SpecialModulePolicy};
use crate::output::{NoOutputIntegrationPolicy, OutputIntegrationPolicy};
use crate::source_policy::{NoSourceIntegrationPolicy, SourceIntegrationPolicy};

pub struct BuildEngine {
    project_root: PathBuf,
    config: BuildConfig,
    providers: ProviderPipeline,
    policies: DriverPolicies,
}

impl BuildEngine {
    pub fn builder(project_root: impl Into<PathBuf>) -> BuildEngineBuilder {
        BuildEngineBuilder::new(project_root.into())
    }

    pub fn discover(&self, entry: impl AsRef<Path>) -> Result<(Bundler, BuildUpdate), String> {
        let entry = if entry.as_ref().is_absolute() {
            entry.as_ref().to_path_buf()
        } else {
            self.project_root.join(entry)
        };
        Bundler::discover_with_driver_policies(
            &entry,
            &self.config,
            self.providers.clone(),
            DriverPolicies {
                compiler: Arc::clone(&self.policies.compiler),
                special_modules: Arc::clone(&self.policies.special_modules),
                runtime: Arc::clone(&self.policies.runtime),
                output: Arc::clone(&self.policies.output),
                source: Arc::clone(&self.policies.source),
            },
        )
    }
}

pub struct BuildEngineBuilder {
    project_root: PathBuf,
    config: BuildConfig,
    environment: Environment,
    providers: Vec<Box<dyn ModuleProvider>>,
    compiler: Arc<dyn ModuleCompiler>,
    special_modules: Arc<dyn SpecialModulePolicy>,
    runtime: Arc<dyn RuntimeIntegrationPolicy>,
    output: Arc<dyn OutputIntegrationPolicy>,
    source: Arc<dyn SourceIntegrationPolicy>,
}

impl BuildEngineBuilder {
    fn new(project_root: PathBuf) -> Self {
        Self {
            project_root,
            config: BuildConfig::default(),
            environment: Environment {
                name: "default".into(),
                platform: Platform::Neutral,
                mode: BuildMode::Production,
            },
            providers: Vec::new(),
            compiler: Arc::new(CoreModuleCompiler),
            special_modules: Arc::new(NoSpecialModulePolicy),
            runtime: Arc::new(NoRuntimeIntegrationPolicy),
            output: Arc::new(NoOutputIntegrationPolicy),
            source: Arc::new(NoSourceIntegrationPolicy),
        }
    }

    pub fn environment(mut self, environment: Environment) -> Self {
        self.environment = environment;
        self
    }

    pub fn config(mut self, config: BuildConfig) -> Self {
        self.config = config;
        self
    }

    pub fn provider(mut self, provider: impl ModuleProvider + 'static) -> Self {
        self.providers.push(Box::new(provider));
        self
    }

    pub fn compiler(mut self, compiler: Arc<dyn ModuleCompiler>) -> Self {
        self.compiler = compiler;
        self
    }

    pub fn source_policy(mut self, policy: Arc<dyn SourceIntegrationPolicy>) -> Self {
        self.source = policy;
        self
    }

    pub fn runtime_policy(mut self, policy: Arc<dyn RuntimeIntegrationPolicy>) -> Self {
        self.runtime = policy;
        self
    }

    pub fn special_module_policy(mut self, policy: Arc<dyn SpecialModulePolicy>) -> Self {
        self.special_modules = policy;
        self
    }

    pub fn output_policy(mut self, policy: Arc<dyn OutputIntegrationPolicy>) -> Self {
        self.output = policy;
        self
    }

    pub fn build(mut self) -> Result<BuildEngine, String> {
        self.project_root = self.project_root.canonicalize().map_err(|error| {
            format!(
                "cannot open project root {}: {error}",
                self.project_root.display()
            )
        })?;
        self.config.target = match self.environment.platform {
            Platform::Browser => diffpack_core::transform::Target::Client,
            Platform::Node | Platform::Neutral => diffpack_core::transform::Target::Server,
        };
        self.config.hmr = self.environment.mode == BuildMode::Development;
        self.config.source_policy = Arc::clone(&self.source);
        Ok(BuildEngine {
            project_root: self.project_root,
            config: self.config,
            providers: ProviderPipeline::new(self.providers),
            policies: DriverPolicies {
                compiler: self.compiler,
                special_modules: self.special_modules,
                runtime: self.runtime,
                output: self.output,
                source: self.source,
            },
        })
    }
}
