/// Backend-agnostic trait for executing compiled tensor computations.
///
/// Each backend compiles from Graph -> its own internal representation,
/// then implements this trait to run it.
pub trait TensorRuntime {
    /// Human-readable backend name (e.g. "wgpu", "wasm", "arm").
    fn backend_name(&self) -> &str;

    /// Run with concrete inputs. Returns flat f32 output.
    fn run(&mut self, inputs: &[&[f32]], output_size: usize) -> Vec<f32> {
        self.run_with_dim_params(&[], inputs, output_size)
    }

    /// Run with symbolic dimension parameters and concrete inputs.
    fn run_with_dim_params(
        &mut self,
        dim_param_values: &[u32],
        inputs: &[&[f32]],
        output_size: usize,
    ) -> Vec<f32>;
}
