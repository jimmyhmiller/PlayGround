// LOCAL product catalog (no fetch, no network) — the ISR listing and the SSG product
// pages both read this static array.
export interface Product {
  id: string;
  name: string;
  price: number;
}

export const PRODUCTS: Product[] = [
  { id: "a", name: "Widget A", price: 10 },
  { id: "b", name: "Widget B", price: 20 },
];

export function findProduct(id: string): Product | undefined {
  return PRODUCTS.find((p) => p.id === id);
}
