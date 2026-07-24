// A trivially code-split module. Its dynamic import forces the client build onto
// the registry runtime (and therefore installs the RSC `__webpack_*` seam), rather
// than the single-chunk scope-hoisted output — exactly as the seam fixture does.
export const value = "lazy-loaded";
