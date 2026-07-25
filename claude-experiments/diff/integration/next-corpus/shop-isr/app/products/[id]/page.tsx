// `products/[id]` — a dynamic segment WITH `generateStaticParams` (over the local
// catalog) and NO request read → classified `ssg`. `export const dynamicParams = true`
// makes it explicit that unlisted ids render on demand (the default).
import { PRODUCTS, findProduct } from "../../products";

export function generateStaticParams() {
  return PRODUCTS.map((p) => ({ id: p.id }));
}

export const dynamicParams = true;

export default async function ProductPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const product = findProduct(id);
  return (
    <main id="product">
      <h1>product: {id}</h1>
      <p>{product ? `${product.name} ($${product.price})` : "unknown product"}</p>
    </main>
  );
}
