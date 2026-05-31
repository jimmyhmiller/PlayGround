while (a) {
  if (b)
    continue;
  c;
}

label0: while (a) {
  b;
  label1: while (d)
    if (c)
      continue label0;
}
