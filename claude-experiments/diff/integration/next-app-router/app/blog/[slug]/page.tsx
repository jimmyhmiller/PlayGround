// A dynamic app-router route: `app/blog/[slug]/page.tsx` matches `/blog/<slug>`,
// with the captured segment delivered as the `params` prop (a Promise, per the
// Next 16 shape). It also reads the request `cookies()` (next/headers) inside this
// async Server Component — proving the per-request context (AsyncLocalStorage) is
// established by the react-server render — and embeds a `"use client"` island that
// reads the same segment via `useParams()`. `next build` accepts this unchanged;
// diffpack's adapter parses the `[slug]` directory into a Dynamic segment, matches
// the request path per-request, and passes `Promise.resolve({ slug })` into it.
//
// It ALSO exports `generateStaticParams` — yet reading `cookies()` at the top opts
// the WHOLE route into dynamic rendering (per Next's own docs: a request-state read
// forces dynamic regardless of generateStaticParams). This is the classification
// PRECEDENCE exemplar: both `next build` and diffpack classify `/blog/[slug]` as
// Dynamic (ƒ) despite the generateStaticParams export — the route is served
// per-request (cookies read live), NOT prerendered to static files. Contrast
// `/products/[id]`, which has generateStaticParams and NO request read, so it is
// SSG-prerendered (one static .html per param). Removing the `cookies()` read here
// would (correctly) flip this route to SSG — the request read is the deciding factor.
import { cookies } from "next/headers";
import { SlugBadge } from "./SlugBadge";

// Enumerated at build for routing, but the cookies() read below forces this route to
// render per-request — so diffpack (like next build) skips it during static prerender
// and records it Dynamic with a reason naming the precedence.
export function generateStaticParams() {
  return [{ slug: "hello" }, { slug: "world" }];
}

export default async function Post({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const cookieStore = await cookies();
  const theme = cookieStore.get("theme")?.value ?? "none";
  return (
    <main id="post">
      post: {slug}
      <span id="theme">theme: {theme}</span>
      <SlugBadge />
    </main>
  );
}
