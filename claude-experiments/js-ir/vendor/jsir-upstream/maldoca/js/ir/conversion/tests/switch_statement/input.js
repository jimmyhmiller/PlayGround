switch (a) {
  case 0:
    body0;
    break;
  case 1:
    body1;
  default:
    break;
}

switch (a) {
  case f():
    body0;
  default:
    break;
  case 1+1:
    body1;
    break;
}

switch (a) {}

switch (a) {
  case 0:
}
