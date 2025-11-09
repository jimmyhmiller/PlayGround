#!/bin/bash
# Quick progress check - shows how many files are in each category

echo "📊 Current Parser Progress"
echo "========================================================================"

echo ""
echo "✅ PERFECT MATCHES:"
matches=$(grep "MATCHES official parser" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
echo "   $matches files produce identical ASTs"

echo ""
echo "🔸 PARSE BUT HAVE ISSUES:"
underscore=$(grep "underscore" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
dotbang=$(grep "dot/bang" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
other_ast=$(grep "other AST" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
echo "   $underscore files - underscore handling (_)"
echo "   $dotbang files - dot/bang operators (.!)"
echo "   $other_ast files - other AST differences"

echo ""
echo "❌ PARSE ERRORS - Missing Features:"
import_export=$(grep -E "(import|provide)" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
prelude=$(grep "lang/provide-types" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
type_alias=$(grep "type alias" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
runtime=$(grep -E "(cases block|table literal|ask expression|lambda block)" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
unknown=$(grep "unknown" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')

echo "   $import_export files - import/export features"
echo "   $prelude files - prelude (#lang, provide-types)"
echo "   $type_alias files - type aliases"
echo "   $runtime files - runtime features (cases/table/ask/lambda)"
echo "   $unknown files - unknown features"

echo ""
echo "💥 CRASHES:"
crashes=$(grep "stack overflow" bulk_test_results/failing_files.txt 2>/dev/null | wc -l | tr -d ' ')
echo "   $crashes files - stack overflow"

echo ""
echo "========================================================================"
total=299
working=$matches
percent=$(( working * 100 / total ))
echo "TOTAL: $working / $total files produce correct ASTs ($percent%)"
echo ""
echo "Run './reannotate.sh' to update after making changes"
echo "Run 'python3 print_summary.py' for detailed breakdown"
