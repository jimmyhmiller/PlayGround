import { seed } from "../../../../src/world/index.js";

export default seed("catalog-basic", {
  users: {
    shopper: { email: "shopper@test.dev", role: "customer" },
  },
  products: {
    "blue-widget": { name: "Blue Widget", price: 1999, stock: 12 },
    "red-widget": { name: "Red Widget", price: 2499, stock: 3 },
    "green-widget": { name: "Green Widget", price: 999, stock: 0 },
  },
});
