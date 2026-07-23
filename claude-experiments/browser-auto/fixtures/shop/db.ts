/** In-memory database for the fixture shop. Module-level singleton so the
 * server, the world adapter, and in-process tests all see the same state. */

export interface User {
  id: number;
  email: string;
  role: string;
}

export interface Product {
  id: number;
  name: string;
  price: number; // cents
  stock: number;
}

export interface CartLine {
  productId: number;
  qty: number;
}

export interface ShopDb {
  users: Map<string, User>;
  products: Map<string, Product>;
  cart: CartLine[];
  nextId: number;
}

export const db: ShopDb = {
  users: new Map(),
  products: new Map(),
  cart: [],
  nextId: 1,
};

export function resetDb(): void {
  db.users.clear();
  db.products.clear();
  db.cart = [];
  db.nextId = 1;
}

export function serializeDb(): string {
  return JSON.stringify({
    users: [...db.users.entries()],
    products: [...db.products.entries()],
    cart: db.cart,
    nextId: db.nextId,
  });
}

export function restoreDb(serialized: string): void {
  const data = JSON.parse(serialized) as {
    users: Array<[string, User]>;
    products: Array<[string, Product]>;
    cart: CartLine[];
    nextId: number;
  };
  db.users = new Map(data.users);
  db.products = new Map(data.products);
  db.cart = data.cart;
  db.nextId = data.nextId;
}
