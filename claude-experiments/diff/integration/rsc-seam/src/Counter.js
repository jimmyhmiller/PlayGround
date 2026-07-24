"use client";

// A real client component. In the CLIENT build this ships as real code and is
// registered in diffpack's registry; the RSC seam must resolve a client reference
// ("<moduleId>#Counter") back to THIS function.
export function Counter() {
  return "REAL-COUNTER";
}

export const Label = "REAL-LABEL";
