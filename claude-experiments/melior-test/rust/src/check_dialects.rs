use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Location, Module, operation::OperationBuilder},
    utility::register_all_dialects,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    
    // Register all available dialects
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    // Allow unregistered dialects for Transform and PDL
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    let module = Module::new(location);
    
    // Test creating a transform operation
    let transform_op = OperationBuilder::new("transform.sequence", location)
        .add_regions([Default::default()])
        .build()?;
        
    println!("Transform operation created successfully: {}", transform_op);
    
    // Test creating a PDL pattern operation
    let pdl_pattern = OperationBuilder::new("pdl.pattern", location)
        .add_attributes(&[(
            melior::ir::Identifier::new(&context, "benefit"),
            melior::ir::attribute::IntegerAttribute::new(
                melior::ir::r#type::IntegerType::new(&context, 64).into(),
                1
            ).into()
        )])
        .add_regions([Default::default()])
        .build()?;
        
    println!("PDL pattern created successfully: {}", pdl_pattern);
    
    println!("Transform + PDL integration ready!");
    
    Ok(())
}