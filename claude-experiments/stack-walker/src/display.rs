//! Display formatting for stack traces
//!
//! This module provides traits and implementations for formatting
//! stack traces in various output formats.

use crate::frame::{Frame, StackTrace, UnwindMethod};
use crate::symbol::{SymbolInfo, SymbolResolver};
use std::fmt::Write;

/// Trait for formatting stack frames
pub trait FrameFormatter {
    /// Format a single frame
    fn format_frame(
        &self,
        frame: &Frame,
        symbol: Option<&SymbolInfo>,
        output: &mut String,
    ) -> std::fmt::Result;

    /// Format an entire stack trace
    fn format_trace<R: SymbolResolver>(
        &self,
        trace: &StackTrace,
        resolver: Option<&R>,
        output: &mut String,
    ) -> std::fmt::Result {
        for frame in trace.iter() {
            let symbol = resolver.and_then(|r| r.resolve(frame.lookup_address()));
            self.format_frame(frame, symbol.as_ref(), output)?;
            writeln!(output)?;
        }

        if trace.truncated {
            if let Some(reason) = &trace.truncation_reason {
                writeln!(output, "... (truncated: {})", reason)?;
            } else {
                writeln!(output, "... (truncated)")?;
            }
        }

        Ok(())
    }

    /// Format a trace to a String
    fn trace_to_string<R: SymbolResolver>(
        &self,
        trace: &StackTrace,
        resolver: Option<&R>,
    ) -> String {
        let mut output = String::new();
        let _ = self.format_trace(trace, resolver, &mut output);
        output
    }
}

/// Plain text backtrace-style formatter
///
/// Output format:
/// ```text
/// # 0 0x0000000000001234 in function_name+0x10 at /path/file.rs:42
/// # 1 0x0000000000005678 in caller_function at /path/other.rs:100
/// ```
#[derive(Debug, Clone)]
pub struct BacktraceFormatter {
    /// Show frame addresses
    pub show_addresses: bool,
    /// Show unwind method
    pub show_unwind_method: bool,
    /// Show source location
    pub show_source: bool,
    /// Show frame pointer values
    pub show_frame_pointers: bool,
}

impl Default for BacktraceFormatter {
    fn default() -> Self {
        Self {
            show_addresses: true,
            show_unwind_method: false,
            show_source: true,
            show_frame_pointers: false,
        }
    }
}

impl BacktraceFormatter {
    /// Create a new backtrace formatter with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a verbose formatter showing all details
    pub fn verbose() -> Self {
        Self {
            show_addresses: true,
            show_unwind_method: true,
            show_source: true,
            show_frame_pointers: true,
        }
    }

    /// Create a minimal formatter
    pub fn minimal() -> Self {
        Self {
            show_addresses: false,
            show_unwind_method: false,
            show_source: true,
            show_frame_pointers: false,
        }
    }
}

impl FrameFormatter for BacktraceFormatter {
    fn format_frame(
        &self,
        frame: &Frame,
        symbol: Option<&SymbolInfo>,
        output: &mut String,
    ) -> std::fmt::Result {
        // Frame index
        write!(output, "#{:>2}", frame.index)?;

        // Address
        if self.show_addresses {
            write!(output, " 0x{:016x}", frame.raw_address())?;
        }

        // Symbol name
        if let Some(sym) = symbol {
            if let Some(name) = sym.display_name() {
                write!(output, " in {}", name)?;

                // Offset within function
                if let Some(offset) = sym.offset_in_function(frame.lookup_address()) {
                    if offset > 0 {
                        write!(output, "+0x{:x}", offset)?;
                    }
                }
            } else {
                write!(output, " in <unknown>")?;
            }

            // Source location
            if self.show_source {
                if let Some(loc) = sym.location() {
                    write!(output, " at {}", loc)?;
                }
            }
        } else {
            write!(output, " in <unknown>")?;
        }

        // Frame pointer
        if self.show_frame_pointers {
            if let Some(fp) = frame.frame_pointer {
                write!(output, " [fp=0x{:x}]", fp)?;
            }
        }

        // Unwind method
        if self.show_unwind_method {
            let method = match frame.unwind_method {
                UnwindMethod::InitialFrame => "initial",
                UnwindMethod::FramePointer => "fp",
            };
            write!(output, " ({})", method)?;
        }

        // Reliability indicator
        if !frame.is_reliable {
            write!(output, " (?)")?;
        }

        Ok(())
    }
}

/// Compact single-line formatter
///
/// Output format:
/// ```text
/// function_name -> caller -> caller2
/// ```
#[derive(Debug, Clone, Default)]
pub struct CompactFormatter {
    /// Separator between frames
    pub separator: String,
    /// Maximum frames to show (0 = unlimited)
    pub max_frames: usize,
}

impl CompactFormatter {
    /// Create a new compact formatter
    pub fn new() -> Self {
        Self {
            separator: " -> ".into(),
            max_frames: 0,
        }
    }

    /// Set the separator
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    /// Set maximum frames
    pub fn with_max_frames(mut self, max: usize) -> Self {
        self.max_frames = max;
        self
    }
}

impl FrameFormatter for CompactFormatter {
    fn format_frame(
        &self,
        frame: &Frame,
        symbol: Option<&SymbolInfo>,
        output: &mut String,
    ) -> std::fmt::Result {
        if let Some(sym) = symbol.and_then(|s| s.display_name()) {
            write!(output, "{}", sym)
        } else {
            write!(output, "0x{:x}", frame.raw_address())
        }
    }

    fn format_trace<R: SymbolResolver>(
        &self,
        trace: &StackTrace,
        resolver: Option<&R>,
        output: &mut String,
    ) -> std::fmt::Result {
        let frames = if self.max_frames > 0 {
            trace.frames.iter().take(self.max_frames)
        } else {
            trace.frames.iter().take(usize::MAX)
        };

        let mut first = true;
        for frame in frames {
            if !first {
                write!(output, "{}", self.separator)?;
            }
            first = false;

            let symbol = resolver.and_then(|r| r.resolve(frame.lookup_address()));
            self.format_frame(frame, symbol.as_ref(), output)?;
        }

        if trace.truncated && self.max_frames == 0 {
            write!(output, "{}...", self.separator)?;
        }

        Ok(())
    }
}

/// JSON formatter
#[derive(Debug, Clone, Default)]
pub struct JsonFormatter {
    /// Pretty print with indentation
    pub pretty: bool,
}

impl JsonFormatter {
    /// Create a new JSON formatter
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a pretty-printing formatter
    pub fn pretty() -> Self {
        Self { pretty: true }
    }
}

impl FrameFormatter for JsonFormatter {
    fn format_frame(
        &self,
        frame: &Frame,
        symbol: Option<&SymbolInfo>,
        output: &mut String,
    ) -> std::fmt::Result {
        write!(output, "{{")?;
        write!(output, "\"index\":{},", frame.index)?;
        write!(output, "\"address\":\"0x{:016x}\",", frame.raw_address())?;
        write!(output, "\"sp\":\"0x{:016x}\",", frame.stack_pointer)?;

        if let Some(fp) = frame.frame_pointer {
            write!(output, "\"fp\":\"0x{:016x}\",", fp)?;
        }

        write!(output, "\"reliable\":{}", frame.is_reliable)?;

        if let Some(sym) = symbol {
            if let Some(name) = &sym.name {
                write!(output, ",\"symbol\":\"{}\"", escape_json(name))?;
            }
            if let Some(demangled) = &sym.demangled_name {
                write!(output, ",\"demangled\":\"{}\"", escape_json(demangled))?;
            }
            if let Some(file) = &sym.file {
                write!(output, ",\"file\":\"{}\"", escape_json(file))?;
            }
            if let Some(line) = sym.line {
                write!(output, ",\"line\":{}", line)?;
            }
            if let Some(col) = sym.column {
                write!(output, ",\"column\":{}", col)?;
            }
        }

        write!(output, "}}")
    }

    fn format_trace<R: SymbolResolver>(
        &self,
        trace: &StackTrace,
        resolver: Option<&R>,
        output: &mut String,
    ) -> std::fmt::Result {
        let indent = if self.pretty { "  " } else { "" };
        let newline = if self.pretty { "\n" } else { "" };

        write!(output, "{{{}", newline)?;
        write!(output, "{}\"frames\": [{}", indent, newline)?;

        for (i, frame) in trace.frames.iter().enumerate() {
            if i > 0 {
                write!(output, ",{}", newline)?;
            }
            if self.pretty {
                write!(output, "    ")?;
            }

            let symbol = resolver.and_then(|r| r.resolve(frame.lookup_address()));
            self.format_frame(frame, symbol.as_ref(), output)?;
        }

        write!(output, "{}", newline)?;
        write!(output, "{}],{}", indent, newline)?;
        write!(output, "{}\"truncated\": {},{}", indent, trace.truncated, newline)?;
        write!(output, "{}\"frame_count\": {}{}", indent, trace.len(), newline)?;
        write!(output, "}}")
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Simple address-only formatter
///
/// Output format:
/// ```text
/// 0x1234
/// 0x5678
/// 0x9abc
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AddressFormatter;

impl FrameFormatter for AddressFormatter {
    fn format_frame(
        &self,
        frame: &Frame,
        _symbol: Option<&SymbolInfo>,
        output: &mut String,
    ) -> std::fmt::Result {
        write!(output, "0x{:016x}", frame.raw_address())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameAddress, UnwindMethod};
    use crate::symbol::MapSymbolResolver;

    fn make_test_trace() -> StackTrace {
        let mut trace = StackTrace::new();
        trace.push(Frame::new(
            0,
            FrameAddress::InstructionPointer(0x1000),
            0x7fff0000,
            Some(0x7fff0010),
            UnwindMethod::InitialFrame,
        ));
        trace.push(Frame::new(
            1,
            FrameAddress::ReturnAddress(0x2000),
            0x7fff0020,
            Some(0x7fff0040),
            UnwindMethod::FramePointer,
        ));
        trace
    }

    #[test]
    fn test_backtrace_formatter() {
        let trace = make_test_trace();
        let formatter = BacktraceFormatter::new();

        let output = formatter.trace_to_string::<MapSymbolResolver>(&trace, None);

        assert!(output.contains("# 0"));
        assert!(output.contains("0x0000000000001000"));
        assert!(output.contains("# 1"));
        assert!(output.contains("0x0000000000002000"));
    }

    #[test]
    fn test_backtrace_with_symbols() {
        let trace = make_test_trace();
        let formatter = BacktraceFormatter::new();

        let mut resolver = MapSymbolResolver::new();
        resolver.add(0x1000, SymbolInfo {
            name: Some("main".into()),
            file: Some("/src/main.rs".into()),
            line: Some(10),
            ..Default::default()
        });

        let output = formatter.trace_to_string(&trace, Some(&resolver));

        assert!(output.contains("main"));
        assert!(output.contains("/src/main.rs:10"));
    }

    #[test]
    fn test_compact_formatter() {
        let trace = make_test_trace();
        let formatter = CompactFormatter::new();

        // Use RangeSymbolResolver since lookup_address() subtracts 1 from return addresses
        let mut resolver = crate::symbol::RangeSymbolResolver::new();
        resolver.add(0x1000, 0x100, SymbolInfo::with_name("main"));
        resolver.add(0x1f00, 0x200, SymbolInfo::with_name("caller")); // 0x1fff falls in this range

        let output = formatter.trace_to_string(&trace, Some(&resolver));

        assert_eq!(output, "main -> caller");
    }

    #[test]
    fn test_json_formatter() {
        let trace = make_test_trace();
        let formatter = JsonFormatter::new();

        let output = formatter.trace_to_string::<MapSymbolResolver>(&trace, None);

        assert!(output.contains("\"frames\""));
        assert!(output.contains("\"index\":0"));
        // Check for truncated field (may have space after colon)
        assert!(output.contains("\"truncated\""));
        assert!(output.contains("false"));
    }
}
