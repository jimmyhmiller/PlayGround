// `next/server` shim (diffpack next app-router adapter).
function cookieJar(headers, isResponse) {
  return {
    get(name) {
      const raw = headers.get("cookie") || "";
      const hit = raw.split(";").map((c) => c.trim()).find((c) => c.startsWith(name + "="));
      return hit ? { name, value: decodeURIComponent(hit.slice(name.length + 1)) } : undefined;
    },
    getAll() {
      const raw = headers.get("cookie") || "";
      return raw.split(";").map((c) => c.trim()).filter(Boolean).map((c) => {
        const eq = c.indexOf("=");
        return { name: c.slice(0, eq), value: decodeURIComponent(c.slice(eq + 1)) };
      });
    },
    set(name, value, opts) {
      const parts = [`${name}=${encodeURIComponent(typeof name === "object" ? name.value : value)}`];
      const o = typeof name === "object" ? name : opts || {};
      if (o.path) parts.push(`Path=${o.path}`);
      if (o.maxAge != null) parts.push(`Max-Age=${o.maxAge}`);
      if (o.httpOnly) parts.push("HttpOnly");
      if (o.secure) parts.push("Secure");
      if (o.sameSite) parts.push(`SameSite=${o.sameSite}`);
      headers.append("set-cookie", parts.join("; "));
      return this;
    },
    delete(name) {
      headers.append("set-cookie", `${name}=; Max-Age=0`);
      return this;
    },
  };
}

export class NextResponse extends Response {
  get cookies() {
    return cookieJar(this.headers, true);
  }
  static next(init) {
    const headers = new Headers(init && init.headers);
    // Request-header overrides (NextResponse.next({ request: { headers } })) are
    // encoded for the orchestrator to apply to the downstream render.
    if (init && init.request && init.request.headers) {
      const reqHeaders = new Headers(init.request.headers);
      const names = [];
      for (const [k, v] of reqHeaders) {
        names.push(k);
        headers.set("x-middleware-request-" + k, v);
      }
      headers.set("x-middleware-override-headers", names.join(","));
    }
    headers.set("x-middleware-next", "1");
    return new NextResponse(null, { headers });
  }
  static redirect(url, init) {
    const status = typeof init === "number" ? init : (init && init.status) || 307;
    const headers = new Headers(init && typeof init === "object" ? init.headers : undefined);
    headers.set("location", String(url));
    return new NextResponse(null, { status, headers });
  }
  static rewrite(destination, init) {
    const headers = new Headers(init && init.headers);
    headers.set("x-middleware-rewrite", String(destination));
    return new NextResponse(null, { headers });
  }
  static json(body, init) {
    const headers = new Headers(init && init.headers);
    if (!headers.has("content-type")) headers.set("content-type", "application/json");
    return new NextResponse(JSON.stringify(body), { ...(init || {}), headers });
  }
}

export class NextRequest extends Request {
  constructor(input, init) {
    super(input, init);
    const url = typeof input === "string" ? input : input.url;
    this.nextUrl = new URL(url, "http://localhost");
  }
  get cookies() {
    return cookieJar(this.headers, false);
  }
}

export default NextResponse;
export const userAgent = (request) => ({ ua: (request.headers.get("user-agent") || "") });
