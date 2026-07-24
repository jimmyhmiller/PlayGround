// An SSG (statically-generated) app-router route: a plain async Server Component that
// reads its `params` but NO request-scoped state (no cookies/headers/searchParams).
// Because it exports `generateStaticParams`, diffpack (like `next build`) enumerates
// the concrete param sets at build time and PRERENDERS one static `.html` + `.rsc` per
// combo — served with zero per-request render. `next build` accepts this unchanged.
export function generateStaticParams() {
  return [{ id: "a" }, { id: "b" }];
}

export default async function Product({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <main id="product">product: {id}</main>;
}
