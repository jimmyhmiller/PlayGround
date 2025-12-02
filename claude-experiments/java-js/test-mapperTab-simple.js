// Simplified version of the problematic code from mapperTab.js
class C {
  static async mergeObjectAsync(e, t) {
    const r = [];
    for (const e of t) r.push({key: await e.key, value: await e.value});
    return r;
  }
}
