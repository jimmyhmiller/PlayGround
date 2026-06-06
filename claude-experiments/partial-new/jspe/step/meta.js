let META = { code: [], leaders: new Set(), loopHeads: new Set(), loopModified: {} };
module.exports = { setMeta: (m) => { META = m; }, get: () => META };
