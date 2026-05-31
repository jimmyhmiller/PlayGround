// SOURCE:      while (a) {
// SOURCE-NEXT:   if (b) continue;
// SOURCE-NEXT:   c;
// SOURCE-NEXT: }
// SOURCE-NEXT: label0: while (a) {
// SOURCE-NEXT:   b;
// SOURCE-NEXT:   label1: while (d) if (c) continue label0;
// SOURCE-NEXT: }
