const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // MLIR library path configuration - auto-detect or use override
    const mlir_path = b.option([]const u8, "mlir-path", "Path to MLIR installation (optional override)") orelse blk: {
        // Try common installation locations
        const possible_paths = [_][]const u8{
            "/opt/homebrew/opt/llvm", // Homebrew Apple Silicon
            "/usr/local/opt/llvm", // Homebrew Intel Mac
            "/usr/local", // Standard Linux install
            "/usr", // System install
        };

        for (possible_paths) |path| {
            const include_path = b.fmt("{s}/include/mlir-c/IR.h", .{path});
            const file = std.fs.openFileAbsolute(include_path, .{}) catch continue;
            file.close();
            break :blk path;
        }

        // Default fallback
        break :blk "/usr/local";
    };

    const mlir_include_path = b.fmt("{s}/include", .{mlir_path});
    const mlir_lib_path = b.fmt("{s}/lib", .{mlir_path});

    // Helper function to link MLIR to a compile step
    const linkMLIR = struct {
        fn link(step: *std.Build.Step.Compile, include_path: []const u8, lib_path: []const u8) void {
            step.addIncludePath(.{ .cwd_relative = include_path });
            step.addLibraryPath(.{ .cwd_relative = lib_path });

            // MLIR C API libraries
            step.linkSystemLibrary("MLIRCAPIIR");
            step.linkSystemLibrary("MLIRCAPIExecutionEngine");
            step.linkSystemLibrary("MLIRCAPIRegisterEverything");
            step.linkSystemLibrary("MLIRCAPIConversion");
            step.linkSystemLibrary("MLIRCAPITransforms");

            // MLIR libraries
            step.linkSystemLibrary("MLIRExecutionEngine");
            step.linkSystemLibrary("MLIRExecutionEngineUtils");
            step.linkSystemLibrary("MLIR");

            // Monolithic LLVM library (contains all LLVM components)
            step.linkSystemLibrary("LLVM");

            step.linkLibCpp();
            step.linkLibC();
        }
    }.link;

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("mlir_lisp", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
    });
    // Add MLIR include path to the module so ZLS can find headers
    mod.addIncludePath(.{ .cwd_relative = mlir_include_path });

    // Create a module for main.zig so it can be imported in tests
    const main_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "mlir_lisp", .module = mod },
        },
    });

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // business logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "mlir_lisp",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "mlir_lisp" is the name you will use in your source code to
                // import this module (e.g. `@import("mlir_lisp")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });

    // Link MLIR C library
    linkMLIR(exe, mlir_include_path, mlir_lib_path);

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    linkMLIR(mod_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    linkMLIR(exe_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // Creates a test executable for test/main_test.zig with access to the main module
    const main_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/main_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "main", .module = main_mod },
            },
        }),
    });

    // A run step that will run the main test executable.
    const run_main_tests = b.addRunArtifact(main_tests);

    // Creates a test executable for grammar examples
    const grammar_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/grammar_examples_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });

    // A run step that will run the grammar test executable.
    const run_grammar_tests = b.addRunArtifact(grammar_tests);

    // Creates a test executable for MLIR integration
    const mlir_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/mlir_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(mlir_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the MLIR test executable.
    const run_mlir_tests = b.addRunArtifact(mlir_tests);

    // Creates a test executable for parser
    const parser_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/parser_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(parser_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the parser test executable.
    const run_parser_tests = b.addRunArtifact(parser_tests);

    // Creates a test executable for builder
    const builder_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/builder_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(builder_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the builder test executable.
    const run_builder_tests = b.addRunArtifact(builder_tests);

    // Creates a test executable for printer
    const printer_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/printer_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(printer_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the printer test executable.
    const run_printer_tests = b.addRunArtifact(printer_tests);

    // Creates a test executable for printer validation
    const printer_validation_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/printer_validation_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(printer_validation_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the printer validation test executable.
    const run_printer_validation_tests = b.addRunArtifact(printer_validation_tests);

    // Creates a test executable for executor
    const executor_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/executor_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(executor_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the executor test executable.
    const run_executor_tests = b.addRunArtifact(executor_tests);

    // Creates a test executable for type builder
    const type_builder_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/type_builder_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(type_builder_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the type builder test executable.
    const run_type_builder_tests = b.addRunArtifact(type_builder_tests);

    // Creates a test executable for REPL integration tests
    const repl_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/integration/repl_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(repl_tests, mlir_include_path, mlir_lib_path);

    // A run step that will run the REPL test executable.
    const run_repl_tests = b.addRunArtifact(repl_tests);

    // Creates an executable to test the printer manually
    const test_printer_exe = b.addExecutable(.{
        .name = "test_printer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/test_printer.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(test_printer_exe, mlir_include_path, mlir_lib_path);
    b.installArtifact(test_printer_exe);

    const run_test_printer = b.addRunArtifact(test_printer_exe);
    const test_printer_step = b.step("test-printer", "Run printer demo");
    test_printer_step.dependOn(&run_test_printer.step);

    // Creates an executable to test the printer with complex examples
    const test_complex_printer_exe = b.addExecutable(.{
        .name = "test_complex_printer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/test_complex_printer.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(test_complex_printer_exe, mlir_include_path, mlir_lib_path);
    b.installArtifact(test_complex_printer_exe);

    const run_test_complex_printer = b.addRunArtifact(test_complex_printer_exe);
    const test_complex_printer_step = b.step("test-complex-printer", "Run complex printer demo");
    test_complex_printer_step.dependOn(&run_test_complex_printer.step);

    // Creates an executable to test JIT compilation
    const jit_example_exe = b.addExecutable(.{
        .name = "jit_example",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/jit_example.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(jit_example_exe, mlir_include_path, mlir_lib_path);
    b.installArtifact(jit_example_exe);

    const run_jit_example = b.addRunArtifact(jit_example_exe);
    const jit_example_step = b.step("jit-example", "Run JIT compilation demo");
    jit_example_step.dependOn(&run_jit_example.step);

    // Creates an executable to test fibonacci JIT compilation with dynamic function calling
    const fib_jit_example_exe = b.addExecutable(.{
        .name = "fib_jit_example",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/fib_jit_example.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(fib_jit_example_exe, mlir_include_path, mlir_lib_path);
    b.installArtifact(fib_jit_example_exe);

    const run_fib_jit_example = b.addRunArtifact(fib_jit_example_exe);
    const fib_jit_example_step = b.step("fib-jit-example", "Run fibonacci JIT compilation demo");
    fib_jit_example_step.dependOn(&run_fib_jit_example.step);

    // Creates an executable to test round-trip parsing/printing
    // NOTE: Commented out because test_roundtrip_funccall.zig was removed
    // const roundtrip_test_exe = b.addExecutable(.{
    //     .name = "test_roundtrip_funccall",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("test_roundtrip_funccall.zig"),
    //         .target = target,
    //         .optimize = optimize,
    //         .imports = &.{
    //             .{ .name = "mlir_lisp", .module = mod },
    //         },
    //     }),
    // });
    // linkMLIR(roundtrip_test_exe, mlir_include_path, mlir_lib_path);
    // b.installArtifact(roundtrip_test_exe);

    // const run_roundtrip_test = b.addRunArtifact(roundtrip_test_exe);
    // const roundtrip_test_step = b.step("test-roundtrip", "Run round-trip test for func.call");
    // roundtrip_test_step.dependOn(&run_roundtrip_test.step);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_main_tests.step);
    test_step.dependOn(&run_grammar_tests.step);
    test_step.dependOn(&run_mlir_tests.step);
    test_step.dependOn(&run_parser_tests.step);
    test_step.dependOn(&run_builder_tests.step);
    test_step.dependOn(&run_printer_tests.step);
    test_step.dependOn(&run_printer_validation_tests.step);
    test_step.dependOn(&run_executor_tests.step);
    test_step.dependOn(&run_type_builder_tests.step);
    test_step.dependOn(&run_repl_tests.step);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}
