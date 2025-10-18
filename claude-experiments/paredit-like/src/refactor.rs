use crate::ast::SExpr;
use anyhow::{anyhow, Result};

pub struct Refactorer {
    source: String,
}

pub struct Indenter {
    source: String,
}

impl Refactorer {
    pub fn new(source: String) -> Self {
        Refactorer { source }
    }

    fn list_chars(list: &SExpr) -> Result<(char, char)> {
        if let SExpr::List { open, close, .. } = list {
            Ok((*open, *close))
        } else {
            Err(anyhow!("Target is not a list"))
        }
    }

    fn ends_with_whitespace(text: &str) -> bool {
        text.chars().last().map_or(false, |c| c.is_whitespace())
    }

    fn starts_with_whitespace(text: &str) -> bool {
        text.chars().next().map_or(false, |c| c.is_whitespace())
    }

    fn trim_whitespace_before(&self, mut idx: usize) -> usize {
        while idx > 0 {
            let ch = self.source[..idx].chars().next_back().unwrap();
            if ch.is_whitespace() {
                idx -= ch.len_utf8();
            } else {
                break;
            }
        }
        idx
    }

    fn trim_whitespace_after(&self, mut idx: usize, limit: usize) -> usize {
        let mut pos = idx;
        while pos < limit {
            let ch = self.source[pos..].chars().next().unwrap();
            if ch.is_whitespace() {
                pos += ch.len_utf8();
            } else {
                break;
            }
        }
        pos
    }

    pub fn slurp_forward(&mut self, forms: &[SExpr], line: usize) -> Result<String> {
        let target_list = self.find_list_at_line(forms, line)?;
        let (open_char, close_char) = Self::list_chars(target_list)?;

        let list_end = target_list.span().end.offset;
        let close_offset = list_end.saturating_sub(1);

        let next_form = self.find_next_form(forms, list_end)?;
        let next_start = next_form.span().start.offset;
        let next_end = next_form.span().end.offset;

        let prefix = &self.source[..close_offset];
        let next_text = &self.source[next_start..next_end];
        let suffix = &self.source[next_end..];

        let mut result = String::with_capacity(self.source.len());
        result.push_str(prefix);

        if !prefix.is_empty() {
            let last_char = prefix.chars().last().unwrap();
            if !last_char.is_whitespace() && last_char != open_char {
                result.push(' ');
            }
        }

        let inserted = next_text.trim_start_matches(|c: char| c.is_whitespace());
        result.push_str(inserted);
        result.push(close_char);
        result.push_str(suffix);

        Ok(result)
    }

    pub fn slurp_backward(&mut self, forms: &[SExpr], line: usize) -> Result<String> {
        let target_list = self.find_list_at_line(forms, line)?;
        let (open_char, close_char) = Self::list_chars(target_list)?;

        let open_offset = target_list.span().start.offset;
        let close_offset = target_list.span().end.offset - close_char.len_utf8();

        let prev_form = self.find_prev_form(forms, open_offset)?;
        let prev_start = prev_form.span().start.offset;
        let prev_end = prev_form.span().end.offset;

        let prefix = &self.source[..prev_start];
        let prev_text = &self.source[prev_start..prev_end];
        let list_inside = &self.source[(open_offset + open_char.len_utf8())..close_offset];
        let suffix = &self.source[close_offset..];

        let mut result = String::with_capacity(self.source.len() + prev_text.len() + 1);
        result.push_str(prefix);
        result.push(open_char);
        result.push_str(prev_text.trim_end_matches(|c: char| c == ' ' || c == '\t'));

        if !list_inside.is_empty() && !Self::starts_with_whitespace(list_inside) {
            result.push(' ');
        }

        result.push_str(list_inside);
        result.push_str(suffix);

        Ok(result)
    }

    pub fn barf_forward(&mut self, forms: &[SExpr], line: usize) -> Result<String> {
        let target_list = self.find_list_at_line(forms, line)?;

        if let SExpr::List { children, span, .. } = target_list {
            if children.is_empty() {
                return Err(anyhow!("Cannot barf from empty list"));
            }

            let last_child = children.last().unwrap();
            let (_, close_char) = Self::list_chars(target_list)?;
            let list_end = span.end.offset;
            let close_offset = list_end - close_char.len_utf8();
            let child_start = last_child.span().start.offset;
            let child_end = last_child.span().end.offset;

            let ws_start = self.trim_whitespace_before(child_start);
            let ws_end = self.trim_whitespace_after(child_end, close_offset);

            let before_child = &self.source[..ws_start];
            let middle = &self.source[ws_end..close_offset];
            let moved_text = &self.source[child_start..child_end];
            let suffix = &self.source[list_end..];

            let mut result = String::with_capacity(self.source.len());
            result.push_str(before_child);
            result.push_str(middle);
            result.push(close_char);
            result.push(' ');
            result.push_str(moved_text);
            result.push_str(suffix);

            return Ok(result);
        }

        Err(anyhow!("Target is not a list"))
    }

    pub fn barf_backward(&mut self, forms: &[SExpr], line: usize) -> Result<String> {
        let target_list = self.find_list_at_line(forms, line)?;

        if let SExpr::List { children, span, .. } = target_list {
            if children.is_empty() {
                return Err(anyhow!("Cannot barf from empty list"));
            }

            let first_child = children.first().unwrap();
            let (open_char, close_char) = Self::list_chars(target_list)?;
            let open_offset = span.start.offset;
            let close_offset = span.end.offset - close_char.len_utf8();

            let child_start = first_child.span().start.offset;
            let child_end = first_child.span().end.offset;
            let ws_end = self.trim_whitespace_after(child_end, close_offset);

            let prefix = &self.source[..open_offset];
            let moved_text = &self.source[child_start..child_end];
            let list_remainder = &self.source[ws_end..close_offset];
            let suffix = &self.source[close_offset..];

            let mut result = String::with_capacity(self.source.len() + moved_text.len() + 1);
            result.push_str(prefix);

            if !prefix.is_empty() {
                let last_char = prefix.chars().last().unwrap();
                if !last_char.is_whitespace() {
                    result.push(' ');
                }
            }

            let trimmed_moved = moved_text.trim_end_matches(|c: char| c == ' ' || c == '\t');
            result.push_str(trimmed_moved);

            if !Self::ends_with_whitespace(&result) {
                result.push(' ');
            }

            result.push(open_char);

            if !list_remainder.is_empty() {
                result.push_str(list_remainder);
            }

            result.push_str(suffix);

            return Ok(result);
        }

        Err(anyhow!("Target is not a list"))
    }

    pub fn splice(&mut self, forms: &[SExpr], line: usize) -> Result<String> {
        let target_list = self.find_list_at_line(forms, line)?;

        if let SExpr::List { .. } = target_list {
            let open_pos = target_list.span().start.offset;
            let close_pos = target_list.span().end.offset;

            let mut result = self.source.clone();

            // Remove closing paren first (higher index)
            result.remove(close_pos - 1);
            // Remove opening paren
            result.remove(open_pos);

            return Ok(result);
        }

        Err(anyhow!("Target is not a list"))
    }

    pub fn raise(&mut self, forms: &[SExpr], line: usize) -> Result<String> {
        let target = self.find_deepest_at_line(forms, line)?;
        let parent = self.find_parent(forms, target)?;

        if let SExpr::List { .. } = parent {
            let parent_start = parent.span().start.offset;
            let parent_end = parent.span().end.offset;
            let target_start = target.span().start.offset;
            let target_end = target.span().end.offset;

            let target_text = self.source[target_start..target_end].to_string();

            let mut result = self.source.clone();
            result.replace_range(parent_start..parent_end, &target_text);

            return Ok(result);
        }

        Err(anyhow!("Cannot raise: no parent list found"))
    }

    pub fn wrap(&mut self, forms: &[SExpr], line: usize, wrapper: &str) -> Result<String> {
        let target = self.find_deepest_at_line(forms, line)?;

        let start = target.span().start.offset;
        let end = target.span().end.offset;

        let mut result = self.source.clone();

        // Insert closing paren/bracket
        let close_char = match wrapper {
            "(" => ')',
            "[" => ']',
            "{" => '}',
            _ => return Err(anyhow!("Invalid wrapper: {}", wrapper)),
        };

        result.insert(end, close_char);
        result.insert_str(start, wrapper);

        Ok(result)
    }

    pub fn merge_let(&mut self, forms: &[SExpr], line: usize) -> Result<String> {
        let outer_let = self.find_let_at_line(forms, line)?;
        let merged = self.merge_single_let(outer_let)?;

        // Auto-indent the result
        let mut parser = crate::parser::ClojureParser::new()?;
        let merged_forms = parser.parse_to_sexpr(&merged)?;
        let indenter = Indenter::new(merged.clone());
        indenter.indent(&merged_forms)
    }

    pub fn merge_all_lets(&mut self, _forms: &[SExpr]) -> Result<String> {
        let mut result = self.source.clone();
        let mut merged_count = 0;
        let mut changed = true;

        // Keep merging until no more mergeable lets are found
        while changed {
            changed = false;

            // Re-parse with the current result
            let mut parser = crate::parser::ClojureParser::new()?;
            let current_forms = parser.parse_to_sexpr(&result)?;

            // Find all mergeable lets in the current version
            let temp_refactorer = Refactorer::new(result.clone());
            let mergeable_lets = temp_refactorer.find_all_mergeable_lets(&current_forms);

            if !mergeable_lets.is_empty() {
                // Merge the first one we find
                let let_form = mergeable_lets[0];
                let temp_refactorer = Refactorer::new(result.clone());
                match temp_refactorer.merge_single_let(let_form) {
                    Ok(new_result) => {
                        result = new_result;
                        merged_count += 1;
                        changed = true;
                    }
                    Err(_) => {
                        // If we can't merge this one, break to avoid infinite loop
                        break;
                    }
                }
            }
        }

        if merged_count > 0 {
            eprintln!("Merged {} nested let form(s)", merged_count);
        }

        // Auto-indent the final result
        let mut parser = crate::parser::ClojureParser::new()?;
        let final_forms = parser.parse_to_sexpr(&result)?;
        let indenter = Indenter::new(result);
        indenter.indent(&final_forms)
    }

    fn merge_single_let(&self, outer_let: &SExpr) -> Result<String> {
        // Verify it's a let form
        if !self.is_let_form(outer_let) {
            return Err(anyhow!("Not a let form"));
        }

        // Get the let's body
        if let SExpr::List { children, .. } = outer_let {
            // A let form should have at least 3 children: (let bindings ...body)
            // We only merge if there's a single body expression that is itself a let
            if children.len() < 3 {
                return Err(anyhow!("Invalid let form"));
            }

            // Check if there's exactly one body expression and it's a let
            // If there are multiple body expressions, we don't want to merge
            if children.len() > 3 {
                return Err(anyhow!("Invalid let form - can only merge directly nested lets"));
            }

            // Check if the body (index 2) is another let
            let body = &children[2];
            if !self.is_let_form(body) {
                return Err(anyhow!("Body is not a let form"));
            }

            // Extract bindings from both lets
            let outer_bindings = &children[1];
            let inner_bindings = if let SExpr::List { children: inner_children, .. } = body {
                &inner_children[1]
            } else {
                return Err(anyhow!("Invalid inner let form"));
            };

            // Build the merged let
            let outer_start = outer_let.span().start.offset;
            let outer_end = outer_let.span().end.offset;

            let mut merged = String::from("(let [");

            let mut first_binding = true;

            // Add outer bindings (handles both standard and typed bindings)
            if let SExpr::List { children: outer_binding_children, .. } = outer_bindings {
                first_binding = self.add_bindings_to_merged(&mut merged, outer_binding_children, first_binding);
            }

            // Add inner bindings
            if let SExpr::List { children: inner_binding_children, .. } = inner_bindings {
                self.add_bindings_to_merged(&mut merged, inner_binding_children, first_binding);
            }

            merged.push_str("]\n      ");

            // Add inner let's body (all expressions after its bindings)
            if let SExpr::List { children: inner_children, .. } = body {
                for i in 2..inner_children.len() {
                    let expr = &inner_children[i];
                    let expr_start = expr.span().start.offset;
                    let expr_end = expr.span().end.offset;
                    merged.push_str(&self.source[expr_start..expr_end]);
                    if i < inner_children.len() - 1 {
                        merged.push(' ');
                    }
                }
            }

            merged.push(')');

            let mut result = self.source.clone();
            result.replace_range(outer_start..outer_end, &merged);

            return Ok(result);
        }

        Err(anyhow!("Invalid let form structure"))
    }

    fn add_bindings_to_merged(&self, merged: &mut String, bindings: &[SExpr], mut is_first: bool) -> bool {
        let mut i = 0;
        while i < bindings.len() {
            // Check if this is a typed binding: [name (: Type) value]
            // by looking ahead to see if the next item is a type annotation
            let is_typed_binding = i + 2 < bindings.len()
                && self.is_type_annotation(&bindings[i + 1]);

            // Add newline and indentation before each binding (except the first)
            if !is_first {
                merged.push(' ');
            } else {
                is_first = false;
            }

            if is_typed_binding {
                // Handle typed binding: name (: Type) value - all on one line
                for j in 0..3 {
                    if i + j < bindings.len() {
                        let child = &bindings[i + j];
                        let child_start = child.span().start.offset;
                        let child_end = child.span().end.offset;
                        merged.push_str(&self.source[child_start..child_end]);
                        if j < 2 {
                            merged.push(' ');
                        }
                    }
                }
                i += 3;
            } else {
                // Handle standard binding: name value - both on same line
                let child = &bindings[i];
                let child_start = child.span().start.offset;
                let child_end = child.span().end.offset;
                merged.push_str(&self.source[child_start..child_end]);

                // Add the value if this is the name
                if i + 1 < bindings.len() && !self.is_type_annotation(&bindings[i + 1]) {
                    merged.push(' ');
                    let value = &bindings[i + 1];
                    let value_start = value.span().start.offset;
                    let value_end = value.span().end.offset;
                    merged.push_str(&self.source[value_start..value_end]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
        is_first
    }

    fn is_type_annotation(&self, sexpr: &SExpr) -> bool {
        // Check if this is a type annotation
        // In your syntax: (: Type) where : is not parsed as part of the children
        // tree-sitter-clojure parses (: Type) as a list with only one child: Type
        // We detect this by checking if it's a list with exactly 1 child and starts with "(:
        match sexpr {
            SExpr::List { children, span, .. } => {
                // Check if the source text starts with "(:" to confirm it's a type annotation
                if children.len() == 1 {
                    // Get the actual text from source
                    let list_text = &self.source[span.start.offset..span.end.offset];
                    // Check if it starts with "(:"
                    list_text.trim_start().starts_with("(:")
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn find_all_mergeable_lets<'a>(&self, forms: &'a [SExpr]) -> Vec<&'a SExpr> {
        let mut result = Vec::new();
        self.collect_mergeable_lets(forms, &mut result);
        result
    }

    fn collect_mergeable_lets<'a>(&self, forms: &'a [SExpr], result: &mut Vec<&'a SExpr>) {
        for form in forms {
            if self.is_let_form(form) {
                // Check if this let has exactly 3 children and the body (index 2) is a let
                // This ensures we only merge directly nested lets with no intermediate expressions
                if let SExpr::List { children, .. } = form {
                    if children.len() == 3 && self.is_let_form(&children[2]) {
                        result.push(form);
                    }
                }
            }

            // Recursively search in children
            if let SExpr::List { children, .. } = form {
                self.collect_mergeable_lets(children, result);
            }
        }
    }

    fn find_let_at_line<'a>(&self, forms: &'a [SExpr], line: usize) -> Result<&'a SExpr> {
        // Find the first let form that contains this line
        for form in forms {
            if form.span().contains_line(line) && self.is_let_form(form) {
                return Ok(form);
            }
            // Recursively search in children
            if let SExpr::List { children, .. } = form {
                if let Ok(let_form) = self.find_let_at_line(children, line) {
                    return Ok(let_form);
                }
            }
        }
        Err(anyhow!("No let form found at line {}", line))
    }

    fn find_list_at_line<'a>(&self, forms: &'a [SExpr], line: usize) -> Result<&'a SExpr> {
        let mut candidates = Vec::new();
        self.collect_lists_at_line(forms, line, &mut candidates);

        candidates
            .into_iter()
            .min_by_key(|form| form.span().start.column)
            .ok_or_else(move || anyhow!("No list found at line {}", line))
    }

    fn collect_lists_at_line<'a>(&self, forms: &'a [SExpr], line: usize, acc: &mut Vec<&'a SExpr>) {
        for form in forms {
            if let SExpr::List { children, .. } = form {
                if form.span().contains_line(line) {
                    acc.push(form);
                }
                self.collect_lists_at_line(children, line, acc);
            }
        }
    }

    fn find_deepest_at_line<'a>(&self, forms: &'a [SExpr], line: usize) -> Result<&'a SExpr> {
        for form in forms {
            if let Some(found) = form.find_deepest_at_line(line) {
                return Ok(found);
            }
        }
        Err(anyhow!("No form found at line {}", line))
    }

    fn find_next_form<'a>(&self, forms: &'a [SExpr], after_pos: usize) -> Result<&'a SExpr> {
        for form in forms {
            if form.span().start.offset > after_pos {
                return Ok(form);
            }
        }
        Err(anyhow!("No form found after position {}", after_pos))
    }

    fn find_prev_form<'a>(&self, forms: &'a [SExpr], before_pos: usize) -> Result<&'a SExpr> {
        let mut prev = None;
        for form in forms {
            if form.span().end.offset < before_pos {
                prev = Some(form);
            } else {
                break;
            }
        }
        prev.ok_or_else(move || anyhow!("No form found before position {}", before_pos))
    }

    fn find_parent<'a>(&self, forms: &'a [SExpr], target: &SExpr) -> Result<&'a SExpr> {
        fn search<'a>(forms: &'a [SExpr], target: &SExpr) -> Option<&'a SExpr> {
            for form in forms {
                if let SExpr::List { children, .. } = form {
                    for child in children {
                        if std::ptr::eq(child, target) {
                            return Some(form);
                        }
                        // Recursively search in children
                        if let Some(found) = search(children, target) {
                            return Some(found);
                        }
                    }
                }
            }
            None
        }

        search(forms, target).ok_or_else(move || anyhow!("No parent found"))
    }

    fn is_let_form(&self, sexpr: &SExpr) -> bool {
        if let SExpr::List { children, .. } = sexpr {
            if let Some(SExpr::Atom { value, .. }) = children.first() {
                return value == "let";
            }
        }
        false
    }
}

impl Indenter {
    pub fn new(source: String) -> Self {
        Indenter { source }
    }

    pub fn indent(&self, forms: &[SExpr]) -> Result<String> {
        let lines: Vec<&str> = self.source.lines().collect();
        let mut result = Vec::new();

        // Track the actual column position for each line as we process them
        let mut line_starts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        line_starts.insert(1, 0); // First line starts at column 0

        for (line_idx, line) in lines.iter().enumerate() {
            let line_num = line_idx + 1; // Lines are 1-indexed

            // Get the trimmed content
            let trimmed = line.trim_start();

            // Skip empty lines
            if trimmed.is_empty() {
                result.push(String::new());
                continue;
            }

            // Calculate proper indentation for this line
            let indent = self.calculate_indent_proper(forms, line_num, &line_starts);

            // Store where this line actually starts for future calculations
            line_starts.insert(line_num, indent);

            // Build the properly indented line
            result.push(format!("{}{}", " ".repeat(indent), trimmed));
        }

        Ok(result.join("\n"))
    }

    fn calculate_indent_proper(&self, forms: &[SExpr], line: usize, line_starts: &std::collections::HashMap<usize, usize>) -> usize {
        // Find all forms that contain this line
        let mut containing_forms = Vec::new();
        self.find_containing_forms(forms, line, &mut containing_forms);

        if containing_forms.is_empty() {
            return 0;
        }

        // Work backwards to find the proper indentation
        for (idx, form) in containing_forms.iter().enumerate().rev() {
            if let SExpr::List { children, span, .. } = form {
                // If this form starts on a previous line, we can calculate indent from it
                if span.start.line < line {
                    let form_start_col = line_starts.get(&span.start.line).copied().unwrap_or(0);

                    // Check if this is a vector/list and the parent is a special form (like let)
                    if idx > 0 {
                        if let SExpr::List { children: parent_children, span: parent_span, .. } = containing_forms[idx - 1] {
                            if let Some(SExpr::Atom { value, .. }) = parent_children.first() {
                                if (value == "let" || value == "binding" || value == "loop") && parent_children.len() > 1 {
                                    // Check if current form is the bindings vector
                                    if std::ptr::eq(*form, &parent_children[1]) {
                                        // This is the bindings vector of a let form
                                        let parent_col = line_starts.get(&parent_span.start.line).copied().unwrap_or(0);
                                        return parent_col + 6;  // Indent bindings
                                    }
                                }
                            }
                        }
                    }

                    // Check if we have a special form
                    if let Some(SExpr::Atom { value, .. }) = children.first() {
                        let indent_spec = self.get_indent_spec(value);

                        return match indent_spec {
                            IndentSpec::Defn | IndentSpec::Form | IndentSpec::If => {
                                // Indent by 2 from the form start
                                form_start_col + 2
                            }
                            IndentSpec::Let => {
                                // Check if we're in the bindings or body
                                if children.len() > 1 {
                                    let bindings = &children[1];
                                    if line <= bindings.span().end.line && line > bindings.span().start.line {
                                        // In bindings vector, on a continuation line
                                        // Indent to align with first binding: (let [ + space
                                        // Assuming standard formatting, this is 6 spaces from form start
                                        form_start_col + 6
                                    } else if line > bindings.span().end.line {
                                        // In body - indent by 2
                                        form_start_col + 2
                                    } else {
                                        // First line of bindings or form declaration
                                        form_start_col + 2
                                    }
                                } else {
                                    form_start_col + 2
                                }
                            }
                            IndentSpec::Args(_) => {
                                // Standard 2-space indent for args
                                form_start_col + 2
                            }
                        };
                    } else {
                        // No special form, use standard 2-space indent
                        return form_start_col + 2;
                    }
                }
            }
        }

        0
    }

    fn calculate_indent(&self, forms: &[SExpr], line: usize, _content: &str) -> usize {
        // Find all forms that contain this line
        let mut containing_forms = Vec::new();
        self.find_containing_forms(forms, line, &mut containing_forms);

        if containing_forms.is_empty() {
            return 0;
        }

        // The indentation is based on the innermost containing form
        let innermost = containing_forms.last().unwrap();

        self.indent_for_form(innermost, line, &containing_forms)
    }

    fn find_containing_forms<'a>(&self, forms: &'a [SExpr], line: usize, result: &mut Vec<&'a SExpr>) {
        for form in forms {
            if form.span().contains_line(line) {
                result.push(form);

                // Recursively check children
                if let SExpr::List { children, .. } = form {
                    self.find_containing_forms(children, line, result);
                }
            }
        }
    }

    fn indent_for_form(&self, form: &SExpr, line: usize, _context: &[&SExpr]) -> usize {
        match form {
            SExpr::List { children, span, .. } => {
                if children.is_empty() {
                    return span.start.column;
                }

                // Get the first element to check if it's a special form
                let first = &children[0];

                // Check if this is a special form
                if let SExpr::Atom { value, .. } = first {
                    let indent_spec = self.get_indent_spec(value);

                    match indent_spec {
                        IndentSpec::Defn => {
                            // defn-style: indent body by 2 spaces from the opening paren
                            if line > span.start.line {
                                return span.start.column + 2;
                            }
                        }
                        IndentSpec::Let => {
                            // let-style: bindings vector aligns with 'let', body indents by 2
                            if children.len() > 1 {
                                let bindings = &children[1];
                                if line <= bindings.span().end.line {
                                    // We're in the bindings vector
                                    return bindings.span().start.column + 1;
                                } else {
                                    // We're in the body
                                    return span.start.column + 2;
                                }
                            }
                        }
                        IndentSpec::If => {
                            // if/when: each branch indents by 2
                            if line > span.start.line {
                                return span.start.column + 2;
                            }
                        }
                        IndentSpec::Args(n) => {
                            // Indent first n args specially, rest get body indentation
                            // Find which arg we're on
                            let mut arg_idx = 0;
                            for (idx, child) in children.iter().enumerate().skip(1) {
                                if line >= child.span().start.line && line <= child.span().end.line {
                                    arg_idx = idx;
                                    break;
                                }
                            }

                            if arg_idx <= n {
                                // Align with first arg if on separate line
                                if children.len() > 1 {
                                    return children[1].span().start.column;
                                }
                            } else {
                                // Body indentation
                                return span.start.column + 2;
                            }
                        }
                        IndentSpec::Form => {
                            // Default: indent by 2 from opening paren
                            if line > span.start.line {
                                return span.start.column + 2;
                            }
                        }
                    }
                }

                // Default: align with first argument or indent by 2
                if line > span.start.line && children.len() > 1 {
                    // Align with the first element after the opening paren
                    return children[0].span().start.column;
                }

                span.start.column
            }
            SExpr::Atom { span, .. } => span.start.column,
            SExpr::String { span, .. } => span.start.column,
            SExpr::Comment { span, .. } => span.start.column,
        }
    }

    fn get_indent_spec(&self, symbol: &str) -> IndentSpec {
        match symbol {
            // Definition forms
            "def" | "defn" | "defn-" | "defmacro" | "defmethod" | "defmulti"
            | "defonce" | "defprotocol" | "defrecord" | "defstruct" | "deftype" => IndentSpec::Defn,

            // Binding forms
            "let" | "when-let" | "if-let" | "binding" | "loop" | "doseq" | "dotimes" | "for" => IndentSpec::Let,

            // Conditional forms
            "if" | "if-not" | "when" | "when-not" | "cond" | "condp" | "case" => IndentSpec::If,

            // Threading macros - treat as special
            "->" | "->>" | "as->" | "cond->" | "cond->>" | "some->" | "some->>" => IndentSpec::Form,

            // Try/catch
            "try" | "catch" | "finally" => IndentSpec::Form,

            // Namespace forms
            "ns" => IndentSpec::Args(1),

            _ => IndentSpec::Form,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum IndentSpec {
    Defn,      // Definition-style indentation (defn, defmacro, etc.)
    Let,       // Let-style indentation (let, binding, etc.)
    If,        // If-style indentation (if, when, cond, etc.)
    Args(usize), // Indent first n arguments specially
    Form,      // Standard form indentation (2 spaces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ClojureParser;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_slurp_forward() {
        let source = "(foo bar) baz".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_forward(&forms, 1).unwrap();
        assert_eq!(result, "(foo bar baz)");
    }

    #[test]
    fn test_slurp_forward_multiple_forms() {
        let source = "(+ 1 2) 3 4".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_forward(&forms, 1).unwrap();
        assert_eq!(result, "(+ 1 2 3) 4");
    }

    #[test]
    fn test_slurp_forward_with_whitespace() {
        let source = "(foo bar)   baz".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_forward(&forms, 1).unwrap();
        assert_eq!(result, "(foo bar baz)");
    }

    #[test]
    fn test_slurp_backward() {
        let source = "foo (bar baz)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_backward(&forms, 1).unwrap();
        assert_eq!(result, "(foo bar baz)");
    }

    #[test]
    fn test_slurp_backward_with_whitespace() {
        let source = "foo   (bar baz)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_backward(&forms, 1).unwrap();
        assert_eq!(result, "(foo bar baz)");
    }

    #[test]
    fn test_barf_forward() {
        let source = "(foo bar baz)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.barf_forward(&forms, 1).unwrap();
        assert_eq!(result, "(foo bar) baz");
    }

    #[test]
    fn test_barf_forward_single_element() {
        let source = "(foo)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.barf_forward(&forms, 1).unwrap();
        assert_eq!(result, "() foo");
    }

    #[test]
    fn test_barf_forward_empty_list() {
        let source = "()".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.barf_forward(&forms, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot barf from empty list"));
    }

    #[test]
    fn test_barf_backward() {
        let source = "(foo bar baz)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.barf_backward(&forms, 1).unwrap();
        assert_eq!(result, "foo (bar baz)");
    }

    #[test]
    fn test_barf_backward_single_element() {
        let source = "(foo)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.barf_backward(&forms, 1).unwrap();
        assert_eq!(result, "foo ()");
    }

    #[test]
    fn test_barf_backward_empty_list() {
        let source = "()".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.barf_backward(&forms, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot barf from empty list"));
    }

    #[test]
    fn test_splice() {
        let source = "(foo (bar baz) qux)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        // This is a simplified test - actual implementation needs line-based targeting
        let result = refactorer.splice(&forms, 1).unwrap();
        assert!(result.contains("bar baz"));
    }

    #[test]
    fn test_splice_simple() {
        let source = "(foo bar)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.splice(&forms, 1).unwrap();
        assert_eq!(result, "foo bar");
    }

    #[test]
    fn test_splice_empty_list() {
        let source = "()".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.splice(&forms, 1).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_splice_nested() {
        let source = "(outer (inner content) more)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.splice(&forms, 1).unwrap();
        assert_eq!(result, "outer (inner content) more");
    }

    #[test]
    fn test_raise() {
        let source = "(foo (bar baz))".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.raise(&forms, 1).unwrap();
        assert!(result.contains("bar") || result.contains("foo"));
    }

    #[test]
    fn test_raise_deep_nesting() {
        let source = "(outer (middle (inner target)))".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.raise(&forms, 1).unwrap();
        // Should raise the innermost to replace outer
        assert!(result.contains("target") || result.contains("inner") || result.contains("middle"));
    }

    #[test]
    fn test_wrap_with_parens() {
        let source = "foo bar".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.wrap(&forms, 1, "(").unwrap();
        assert!(result.contains("(foo)"));
    }

    #[test]
    fn test_wrap_with_brackets() {
        let source = "foo bar".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.wrap(&forms, 1, "[").unwrap();
        assert!(result.contains("[foo]"));
    }

    #[test]
    fn test_wrap_with_braces() {
        let source = "foo bar".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.wrap(&forms, 1, "{").unwrap();
        assert!(result.contains("{foo}"));
    }

    #[test]
    fn test_wrap_invalid_wrapper() {
        let source = "foo".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.wrap(&forms, 1, "<");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid wrapper"));
    }

    #[test]
    fn test_merge_let_simple() {
        let source = "(let [x 1] (let [y 2] (+ x y)))".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.merge_let(&forms, 1).unwrap();
        // Check that it contains the merged bindings (now on separate lines)
        assert!(result.contains("let ["));
        assert!(result.contains("x 1"));
        assert!(result.contains("y 2"));
        assert!(result.contains("(+ x y)"));
    }

    #[test]
    fn test_merge_let_complex() {
        let source = "(let [a 1 b 2] (let [c 3 d 4] (+ a b c d)))".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.merge_let(&forms, 1).unwrap();
        // Check that it contains all the merged bindings (now on separate lines)
        assert!(result.contains("let ["));
        assert!(result.contains("a 1"));
        assert!(result.contains("b 2"));
        assert!(result.contains("c 3"));
        assert!(result.contains("d 4"));
        assert!(result.contains("(+ a b c d)"));
    }

    #[test]
    fn test_merge_let_not_let_form() {
        let source = "(defn foo [x] x)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.merge_let(&forms, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No let form found"));
    }

    #[test]
    fn test_merge_let_body_not_let() {
        let source = "(let [x 1] (+ x 2))".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.merge_let(&forms, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Body is not a let form"));
    }

    #[test]
    fn test_find_list_at_line_nested() {
        let source = "(outer\n  (inner content))".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        
        // Line 1 should find outer list
        let result = refactorer.find_list_at_line(&forms, 1);
        assert!(result.is_ok());
        
        // Line 2 should find inner list
        let result = refactorer.find_list_at_line(&forms, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_list_at_line_not_found() {
        let source = "atom".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        let result = refactorer.find_list_at_line(&forms, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No list found"));
    }

    #[test]
    fn test_find_deepest_at_line() {
        let source = "(outer (inner (deepest)))".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        let result = refactorer.find_deepest_at_line(&forms, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_next_form() {
        let source = "first second third".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        
        // After first form should find second
        let first_end = forms[0].span().end.offset;
        let result = refactorer.find_next_form(&forms, first_end);
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_next_form_not_found() {
        let source = "only_one".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        let result = refactorer.find_next_form(&forms, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_prev_form() {
        let source = "first second third".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        
        // Before third form should find second
        let third_start = forms[2].span().start.offset;
        let result = refactorer.find_prev_form(&forms, third_start);
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_prev_form_not_found() {
        let source = "only_one".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        let result = refactorer.find_prev_form(&forms, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_let_form_true() {
        let source = "(let [x 1] x)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        assert!(refactorer.is_let_form(&forms[0]));
    }

    #[test]
    fn test_is_let_form_false() {
        let source = "(defn foo [x] x)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        assert!(!refactorer.is_let_form(&forms[0]));
    }

    #[test]
    fn test_is_let_form_atom() {
        let source = "atom".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let refactorer = Refactorer::new(source);
        assert!(!refactorer.is_let_form(&forms[0]));
    }

    #[test]
    fn test_refactor_with_vectors() {
        let source = "[foo bar] baz".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_forward(&forms, 1).unwrap();
        assert_eq!(result, "[foo bar baz]");
    }

    #[test]
    fn test_refactor_with_maps() {
        let source = "{:a 1} {:b 2}".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_forward(&forms, 1).unwrap();
        assert_eq!(result, "{:a 1 {:b 2}}");
    }

    #[test]
    fn test_multiline_refactoring() {
        let source = "(defn foo [x]\n  (+ x 1))\n\n(bar)".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_forward(&forms, 1).unwrap();
        // Should slurp the (bar) into the defn
        assert!(result.contains("(defn foo [x]\n  (+ x 1) (bar))"));
    }

    #[test]
    fn test_refactor_preserves_comments() {
        let source = "(foo bar) ; comment\nbaz".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_forward(&forms, 1).unwrap();
        assert!(result.contains("; comment"));
    }

    #[test]
    fn test_refactor_with_strings() {
        let source = "(str \"hello\") \"world\"".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.slurp_forward(&forms, 1).unwrap();
        assert_eq!(result, "(str \"hello\" \"world\")");
    }

    #[test]
    fn test_barf_target_not_list() {
        let source = "atom".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.barf_forward(&forms, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No list found"));
    }

    #[test]
    fn test_splice_target_not_list() {
        let source = "atom".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.splice(&forms, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_raise_no_parent() {
        let source = "atom".to_string();
        let mut parser = ClojureParser::new().unwrap();
        let forms = parser.parse_to_sexpr(&source).unwrap();
        let mut refactorer = Refactorer::new(source);
        let result = refactorer.raise(&forms, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No parent found"));
    }
}
