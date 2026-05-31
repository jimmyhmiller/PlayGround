// SOURCE:      switch (a) {
// SOURCE-NEXT:   case 0:
// SOURCE-NEXT:     body0;
// SOURCE-NEXT:     break;
// SOURCE-NEXT:   case 1:
// SOURCE-NEXT:     body1;
// SOURCE-NEXT:   default:
// SOURCE-NEXT:     break;
// SOURCE-NEXT: }
// SOURCE-NEXT: switch (a) {
// SOURCE-NEXT:   case f():
// SOURCE-NEXT:     body0;
// SOURCE-NEXT:   default:
// SOURCE-NEXT:     break;
// SOURCE-NEXT:   case 1 + 1:
// SOURCE-NEXT:     body1;
// SOURCE-NEXT:     break;
// SOURCE-NEXT: }
// SOURCE-NEXT: switch (a) {}
// SOURCE-NEXT: switch (a) {
// SOURCE-NEXT:   case 0:
// SOURCE-NEXT: }
