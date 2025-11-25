const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const main_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Show Reader tool
    const show_reader_module = b.createModule(.{
        .root_source_file = b.path("src/show_reader.zig"),
        .target = target,
        .optimize = optimize,
    });
    show_reader_module.addImport("main", main_module);

    const show_reader_exe = b.addExecutable(.{
        .name = "show-reader",
        .root_module = show_reader_module,
    });

    // Add MLIR include paths and libraries for show-reader
    show_reader_exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/include" });
    show_reader_exe.addIncludePath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/include" });
    show_reader_exe.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/lib" });
    show_reader_exe.addLibraryPath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/build" });
    show_reader_exe.linkSystemLibrary("MLIR");
    show_reader_exe.linkSystemLibrary("LLVM");
    show_reader_exe.linkSystemLibrary("MLIRCAPIIR");
    show_reader_exe.linkSystemLibrary("MLIRCAPIFunc");
    show_reader_exe.linkSystemLibrary("MLIRCAPIArith");
    show_reader_exe.linkSystemLibrary("MLIRCAPIControlFlow");
    show_reader_exe.linkSystemLibrary("MLIRCAPISCF");
    show_reader_exe.linkSystemLibrary("MLIRCAPIMemRef");
    show_reader_exe.linkSystemLibrary("MLIRCAPIVector");
    show_reader_exe.linkSystemLibrary("MLIRCAPILLVM");
    show_reader_exe.linkSystemLibrary("mlir-introspection");
    show_reader_exe.linkLibC();
    show_reader_exe.linkLibCpp();

    b.installArtifact(show_reader_exe);

    const run_show_reader = b.addRunArtifact(show_reader_exe);
    run_show_reader.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_show_reader.addArgs(args);
    }

    const show_reader_step = b.step("show-reader", "Show reader output for source code");
    show_reader_step.dependOn(&run_show_reader.step);

    // Show AST tool
    const show_ast_module = b.createModule(.{
        .root_source_file = b.path("src/show_ast.zig"),
        .target = target,
        .optimize = optimize,
    });
    show_ast_module.addImport("main", main_module);

    const show_ast_exe = b.addExecutable(.{
        .name = "show-ast",
        .root_module = show_ast_module,
    });

    // Add MLIR include paths and libraries for show-ast
    show_ast_exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/include" });
    show_ast_exe.addIncludePath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/include" });
    show_ast_exe.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/lib" });
    show_ast_exe.addLibraryPath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/build" });
    show_ast_exe.linkSystemLibrary("MLIR");
    show_ast_exe.linkSystemLibrary("LLVM");
    show_ast_exe.linkSystemLibrary("MLIRCAPIIR");
    show_ast_exe.linkSystemLibrary("MLIRCAPIFunc");
    show_ast_exe.linkSystemLibrary("MLIRCAPIArith");
    show_ast_exe.linkSystemLibrary("MLIRCAPIControlFlow");
    show_ast_exe.linkSystemLibrary("MLIRCAPISCF");
    show_ast_exe.linkSystemLibrary("MLIRCAPIMemRef");
    show_ast_exe.linkSystemLibrary("MLIRCAPIVector");
    show_ast_exe.linkSystemLibrary("MLIRCAPILLVM");
    show_ast_exe.linkSystemLibrary("mlir-introspection");
    show_ast_exe.linkLibC();
    show_ast_exe.linkLibCpp();

    b.installArtifact(show_ast_exe);

    const run_show_ast = b.addRunArtifact(show_ast_exe);
    run_show_ast.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_show_ast.addArgs(args);
    }

    const show_ast_step = b.step("show-ast", "Show AST output for source code");
    show_ast_step.dependOn(&run_show_ast.step);

    // Find overlaps tool
    const find_overlaps_module = b.createModule(.{
        .root_source_file = b.path("src/find_overlaps.zig"),
        .target = target,
        .optimize = optimize,
    });
    find_overlaps_module.addImport("main", main_module);

    const find_overlaps_exe = b.addExecutable(.{
        .name = "find-overlaps",
        .root_module = find_overlaps_module,
    });
    find_overlaps_exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/include" });
    find_overlaps_exe.addIncludePath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/include" });
    find_overlaps_exe.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/lib" });
    find_overlaps_exe.addLibraryPath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/build" });
    find_overlaps_exe.linkSystemLibrary("MLIR");
    find_overlaps_exe.linkSystemLibrary("LLVM");
    find_overlaps_exe.linkSystemLibrary("MLIRCAPIIR");
    find_overlaps_exe.linkSystemLibrary("MLIRCAPIFunc");
    find_overlaps_exe.linkSystemLibrary("MLIRCAPIArith");
    find_overlaps_exe.linkSystemLibrary("MLIRCAPIControlFlow");
    find_overlaps_exe.linkSystemLibrary("MLIRCAPISCF");
    find_overlaps_exe.linkSystemLibrary("MLIRCAPIMemRef");
    find_overlaps_exe.linkSystemLibrary("MLIRCAPIVector");
    find_overlaps_exe.linkSystemLibrary("MLIRCAPILLVM");
    find_overlaps_exe.linkSystemLibrary("mlir-introspection");
    find_overlaps_exe.linkLibC();
    find_overlaps_exe.linkLibCpp();
    b.installArtifact(find_overlaps_exe);

    const run_find_overlaps = b.addRunArtifact(find_overlaps_exe);
    const find_overlaps_step = b.step("find-overlaps", "Find overlapping operations in MLIR dialects");
    find_overlaps_step.dependOn(&run_find_overlaps.step);

    // Tests
    const tests = b.addTest(.{
        .root_module = main_module,
    });

    // Add MLIR include paths and libraries
    tests.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/include" });
    tests.addIncludePath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/include" });
    tests.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/lib" });
    tests.addLibraryPath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/build" });

    // Link MLIR and LLVM libraries
    // Use the main shared libraries which contain all implementations
    tests.linkSystemLibrary("MLIR");
    tests.linkSystemLibrary("LLVM");

    // MLIR C API libraries (for dialect handles and operations)
    tests.linkSystemLibrary("MLIRCAPIIR");
    tests.linkSystemLibrary("MLIRCAPIFunc");
    tests.linkSystemLibrary("MLIRCAPIArith");
    tests.linkSystemLibrary("MLIRCAPIControlFlow");
    tests.linkSystemLibrary("MLIRCAPISCF");
    tests.linkSystemLibrary("MLIRCAPIMemRef");
    tests.linkSystemLibrary("MLIRCAPIVector");
    tests.linkSystemLibrary("MLIRCAPILLVM");

    // C API wrapper library
    tests.linkSystemLibrary("mlir-introspection");

    tests.linkLibC();
    tests.linkLibCpp();

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_tests.step);

    // Integration tests (module path is different, but reuse the same MLIR setup)
    const integration_root = b.createModule(.{
        .root_source_file = b.path("tests/integration_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_root.addImport("main", main_module);

    const integration_tests = b.addTest(.{
        .root_module = integration_root,
    });

    integration_tests.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/include" });
    integration_tests.addIncludePath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/include" });
    integration_tests.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/lib" });
    integration_tests.addLibraryPath(.{ .cwd_relative = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/build" });

    integration_tests.linkSystemLibrary("MLIR");
    integration_tests.linkSystemLibrary("LLVM");
    integration_tests.linkSystemLibrary("MLIRCAPIIR");
    integration_tests.linkSystemLibrary("MLIRCAPIFunc");
    integration_tests.linkSystemLibrary("MLIRCAPIArith");
    integration_tests.linkSystemLibrary("MLIRCAPIControlFlow");
    integration_tests.linkSystemLibrary("MLIRCAPISCF");
    integration_tests.linkSystemLibrary("MLIRCAPIMemRef");
    integration_tests.linkSystemLibrary("MLIRCAPIVector");
    integration_tests.linkSystemLibrary("MLIRCAPILLVM");
    integration_tests.linkSystemLibrary("mlir-introspection");
    integration_tests.linkLibC();
    integration_tests.linkLibCpp();

    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_step = b.step("integration-test", "Run integration tests");
    integration_step.dependOn(&run_integration_tests.step);
}
