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
            "/usr/local", // Standard Linux install (locally built LLVM with ROCm)
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

    // Detect if we're building for Linux (to use correct C++ stdlib)
    const is_linux = target.result.os.tag == .linux;

    // Helper function to link MLIR to a compile step
    const linkMLIR = struct {
        fn link(step: *std.Build.Step.Compile, include_path: []const u8, lib_path: []const u8, linux: bool) void {
            step.addIncludePath(.{ .cwd_relative = include_path });

            // Add library path
            step.addLibraryPath(.{ .cwd_relative = lib_path });

            // MLIR C API libraries
            step.linkSystemLibrary("MLIRCAPIIR");
            step.linkSystemLibrary("MLIRCAPIInterfaces");
            step.linkSystemLibrary("MLIRCAPIExecutionEngine");
            step.linkSystemLibrary("MLIRCAPIRegisterEverything");
            step.linkSystemLibrary("MLIRCAPIConversion");
            step.linkSystemLibrary("MLIRCAPITransforms");
            step.linkSystemLibrary("MLIRCAPIIRDL");
            step.linkSystemLibrary("MLIRCAPITransformDialect");
            step.linkSystemLibrary("MLIRCAPITransformDialectTransforms");

            // MLIR libraries
            step.linkSystemLibrary("MLIRExecutionEngine");
            step.linkSystemLibrary("MLIRExecutionEngineUtils");
            step.linkSystemLibrary("MLIR");

            // GPU/ROCDL libraries for AMD GPU support
            step.linkSystemLibrary("MLIRROCDLToLLVMIRTranslation");
            step.linkSystemLibrary("MLIRROCDLTarget");
            step.linkSystemLibrary("MLIRROCDLDialect");
            step.linkSystemLibrary("MLIRGPUToROCDLTransforms");

            // Monolithic LLVM library (contains all LLVM components)
            step.linkSystemLibrary("LLVM");

            // Link C library
            step.linkLibC();

            // On Linux, LLVM is built against libstdc++ (GNU C++ stdlib)
            // On macOS, use libc++ (LLVM C++ stdlib)
            if (linux) {
                // Workaround for Zig issue #12147:
                // Zig automatically adds -lc++ but we need -lstdc++ on Linux
                // The only way to override this is to provide the .so file directly
                step.addObjectFile(.{ .cwd_relative = "/usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so" });
            } else {
                // On macOS, use LLVM's libc++
                step.linkLibCpp();
            }
        }
    }.link;

    // Helper function to run test executables, working around duplicate rpath issues
    const fixRpathForTest = struct {
        fn fix(builder: *std.Build, test_artifact: *std.Build.Step.Compile, tgt: std.Build.ResolvedTarget, lib_path: []const u8) *std.Build.Step.Run {
            _ = lib_path;

            if (tgt.result.os.tag == .macos) {
                // On macOS, use a wrapper script to fix duplicate rpaths before running
                // This works around Zig issue #24349
                const wrapper = builder.addSystemCommand(&[_][]const u8{
                    "./fix-rpath-and-run.sh",
                });
                wrapper.addArtifactArg(test_artifact);
                wrapper.step.dependOn(&test_artifact.step);

                return wrapper;
            }

            return builder.addRunArtifact(test_artifact);
        }
    }.fix;

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
    linkMLIR(exe, mlir_include_path, mlir_lib_path, is_linux);

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // Workaround for Zig issue #24349: Remove duplicate rpaths on macOS
    // Duplicate rpaths cause dyld to abort on recent macOS versions
    if (target.result.os.tag == .macos) {
        const fix_rpath = b.addSystemCommand(&[_][]const u8{
            "sh",
            "-c",
        });
        const cmd = b.fmt(
            \\# Get unique rpaths and remove duplicates
            \\unique_rpaths=$(otool -l zig-out/bin/mlir_lisp | grep -A 2 LC_RPATH | grep path | awk '{{print $2}}' | sort -u)
            \\all_rpaths=$(otool -l zig-out/bin/mlir_lisp | grep -A 2 LC_RPATH | grep path | awk '{{print $2}}')
            \\# For each rpath, count occurrences and remove duplicates
            \\for rpath in $all_rpaths; do
            \\    count=$(echo "$all_rpaths" | grep -c "^$rpath$")
            \\    if [ "$count" -gt 1 ]; then
            \\        # Remove all but one occurrence
            \\        while [ "$count" -gt 1 ]; do
            \\            install_name_tool -delete_rpath "$rpath" zig-out/bin/mlir_lisp 2>/dev/null || true
            \\            count=$((count - 1))
            \\        done
            \\    fi
            \\done
        , .{});
        fix_rpath.addArg(cmd);
        fix_rpath.step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        b.getInstallStep().dependOn(&fix_rpath.step);
    }

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
    linkMLIR(mod_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the test executable.
    const run_mod_tests = fixRpathForTest(b, mod_tests, target, mlir_lib_path);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    linkMLIR(exe_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the second test executable.
    const run_exe_tests = fixRpathForTest(b, exe_tests, target, mlir_lib_path);

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
    const run_main_tests = fixRpathForTest(b, main_tests, target, mlir_lib_path);

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
    const run_grammar_tests = fixRpathForTest(b, grammar_tests, target, mlir_lib_path);

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
    linkMLIR(mlir_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the MLIR test executable.
    const run_mlir_tests = fixRpathForTest(b, mlir_tests, target, mlir_lib_path);

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
    linkMLIR(parser_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the parser test executable.
    const run_parser_tests = fixRpathForTest(b, parser_tests, target, mlir_lib_path);

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
    linkMLIR(builder_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the builder test executable.
    const run_builder_tests = fixRpathForTest(b, builder_tests, target, mlir_lib_path);

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
    linkMLIR(printer_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the printer test executable.
    const run_printer_tests = fixRpathForTest(b, printer_tests, target, mlir_lib_path);

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
    linkMLIR(printer_validation_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the printer validation test executable.
    const run_printer_validation_tests = fixRpathForTest(b, printer_validation_tests, target, mlir_lib_path);

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
    linkMLIR(executor_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the executor test executable.
    const run_executor_tests = fixRpathForTest(b, executor_tests, target, mlir_lib_path);

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
    linkMLIR(type_builder_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the type builder test executable.
    const run_type_builder_tests = fixRpathForTest(b, type_builder_tests, target, mlir_lib_path);

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
    linkMLIR(repl_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the REPL test executable.
    const run_repl_tests = fixRpathForTest(b, repl_tests, target, mlir_lib_path);
    // REPL tests need the main executable to be installed first
    run_repl_tests.step.dependOn(b.getInstallStep());

    // Creates a test executable for reader print round-trip tests
    const reader_roundtrip_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/reader_print_roundtrip_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(reader_roundtrip_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the reader roundtrip test executable.
    const run_reader_roundtrip_tests = fixRpathForTest(b, reader_roundtrip_tests, target, mlir_lib_path);

    // Creates a test executable for tokenizer
    const tokenizer_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/tokenizer_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(tokenizer_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the tokenizer test executable.
    const run_tokenizer_tests = fixRpathForTest(b, tokenizer_tests, target, mlir_lib_path);

    // Creates a test executable for macro expansion
    const macro_expansion_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/macro_expansion_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(macro_expansion_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the macro expansion test executable.
    const run_macro_expansion_tests = fixRpathForTest(b, macro_expansion_tests, target, mlir_lib_path);

    // Creates a test executable for macro system
    const macro_system_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/macro_system_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(macro_system_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the macro system test executable.
    const run_macro_system_tests = fixRpathForTest(b, macro_system_tests, target, mlir_lib_path);

    // Creates a test executable for operation flattener
    const operation_flattener_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/operation_flattener_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(operation_flattener_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the operation flattener test executable.
    const run_operation_flattener_tests = fixRpathForTest(b, operation_flattener_tests, target, mlir_lib_path);

    // Creates a test executable for op macro
    const op_macro_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/op_macro_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(op_macro_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the op macro test executable.
    const run_op_macro_tests = fixRpathForTest(b, op_macro_tests, target, mlir_lib_path);

    // Creates a test executable for struct access
    const struct_access_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/struct_access_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(struct_access_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the struct access test executable.
    const run_struct_access_tests = fixRpathForTest(b, struct_access_tests, target, mlir_lib_path);

    // Creates a test executable for JIT-compiled macros
    const jit_macro_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/jit_macro_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(jit_macro_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the JIT macro test executable.
    const run_jit_macro_tests = fixRpathForTest(b, jit_macro_tests, target, mlir_lib_path);

    // Creates a test executable for value layout
    const value_layout_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/value_layout_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(value_layout_tests, mlir_include_path, mlir_lib_path, is_linux);

    // A run step that will run the value layout test executable.
    const run_value_layout_tests = fixRpathForTest(b, value_layout_tests, target, mlir_lib_path);

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
    linkMLIR(test_printer_exe, mlir_include_path, mlir_lib_path, is_linux);
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
    linkMLIR(test_complex_printer_exe, mlir_include_path, mlir_lib_path, is_linux);
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
    linkMLIR(jit_example_exe, mlir_include_path, mlir_lib_path, is_linux);
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
    linkMLIR(fib_jit_example_exe, mlir_include_path, mlir_lib_path, is_linux);
    b.installArtifact(fib_jit_example_exe);

    const run_fib_jit_example = b.addRunArtifact(fib_jit_example_exe);
    const fib_jit_example_step = b.step("fib-jit-example", "Run fibonacci JIT compilation demo");
    fib_jit_example_step.dependOn(&run_fib_jit_example.step);

    // Creates an executable to test JIT compilation with runtime symbol resolution
    const temp_testing_mlir_jit_exe = b.addExecutable(.{
        .name = "temp_testing_mlir_jit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/temp_testing_mlir_jit.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "mlir_lisp", .module = mod },
            },
        }),
    });
    linkMLIR(temp_testing_mlir_jit_exe, mlir_include_path, mlir_lib_path, is_linux);
    b.installArtifact(temp_testing_mlir_jit_exe);

    // Workaround for Zig issue #24349: Remove duplicate rpaths on macOS
    if (target.result.os.tag == .macos) {
        const fix_rpath_temp_jit = b.addSystemCommand(&[_][]const u8{
            "sh",
            "-c",
        });
        const cmd_temp_jit = b.fmt(
            \\# Get unique rpaths and remove duplicates
            \\unique_rpaths=$(otool -l zig-out/bin/temp_testing_mlir_jit | grep -A 2 LC_RPATH | grep path | awk '{{print $2}}' | sort -u)
            \\all_rpaths=$(otool -l zig-out/bin/temp_testing_mlir_jit | grep -A 2 LC_RPATH | grep path | awk '{{print $2}}')
            \\# For each rpath, count occurrences and remove duplicates
            \\for rpath in $all_rpaths; do
            \\    count=$(echo "$all_rpaths" | grep -c "^$rpath$")
            \\    if [ "$count" -gt 1 ]; then
            \\        # Remove all but one occurrence
            \\        while [ "$count" -gt 1 ]; do
            \\            install_name_tool -delete_rpath "$rpath" zig-out/bin/temp_testing_mlir_jit 2>/dev/null || true
            \\            count=$((count - 1))
            \\        done
            \\    fi
            \\done
        , .{});
        fix_rpath_temp_jit.addArg(cmd_temp_jit);
        fix_rpath_temp_jit.step.dependOn(&b.addInstallArtifact(temp_testing_mlir_jit_exe, .{}).step);
        b.getInstallStep().dependOn(&fix_rpath_temp_jit.step);
    }

    const run_temp_testing_mlir_jit = b.addRunArtifact(temp_testing_mlir_jit_exe);
    if (b.args) |args| {
        run_temp_testing_mlir_jit.addArgs(args);
    }
    const temp_testing_mlir_jit_step = b.step("temp-jit", "Run temporary JIT testing with runtime symbols");
    temp_testing_mlir_jit_step.dependOn(&run_temp_testing_mlir_jit.step);

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
    // linkMLIR(roundtrip_test_exe, mlir_include_path, mlir_lib_path, is_linux);
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
    test_step.dependOn(&run_reader_roundtrip_tests.step);
    test_step.dependOn(&run_tokenizer_tests.step);
    test_step.dependOn(&run_macro_expansion_tests.step);
    test_step.dependOn(&run_macro_system_tests.step);
    test_step.dependOn(&run_operation_flattener_tests.step);
    test_step.dependOn(&run_op_macro_tests.step);
    test_step.dependOn(&run_struct_access_tests.step);
    test_step.dependOn(&run_jit_macro_tests.step);
    test_step.dependOn(&run_value_layout_tests.step);

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
