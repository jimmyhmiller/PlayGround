import { seed, ref } from "../../../../src/world/index.js";

export default seed("dashboard-basic", {
  users: {
    admin: { name: "Bat Admin", email: "admin@bat.dev", password: "batpass123" },
  },
  customers: {
    "evil-rabbit": { name: "Evil Rabbit", email: "evil@rabbit.com", image_url: "/customers/evil-rabbit.png" },
    delba: { name: "Delba de Oliveira", email: "delba@oliveira.com", image_url: "/customers/delba-de-oliveira.png" },
    lee: { name: "Lee Robinson", email: "lee@robinson.com", image_url: "/customers/lee-robinson.png" },
  },
  invoices: {
    "rabbit-pending": { customer: ref("customers", "evil-rabbit"), amount: 15795, status: "pending", date: "2026-06-05" },
    "delba-paid": { customer: ref("customers", "delba"), amount: 66800, status: "paid", date: "2026-06-12" },
    "lee-paid": { customer: ref("customers", "lee"), amount: 5000, status: "paid", date: "2026-06-20" },
  },
  revenue: {
    Jan: { revenue: 2000 },
    Feb: { revenue: 1800 },
    Mar: { revenue: 2200 },
    Apr: { revenue: 2500 },
    May: { revenue: 2300 },
    Jun: { revenue: 3200 },
  },
});
