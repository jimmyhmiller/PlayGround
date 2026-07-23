/* Fixture shop SPA. Deliberately "modern-flaky": every render waits on a
 * fetch with a random server-side delay, and the add-to-cart toast lives
 * for only 150ms. bat flows must pass against this every single time. */

const app = document.getElementById("app");

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = new Error(`${path} -> ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function setCartCount(n) {
  document.querySelector('[data-testid="cart-count"]').textContent = String(n);
}

function toast(message) {
  const el = document.createElement("div");
  el.className = "toast";
  el.setAttribute("role", "status");
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 150);
}

function money(cents) {
  return `$${(cents / 100).toFixed(2)}`;
}

async function renderProducts() {
  app.innerHTML = "<p>Loading products…</p>";
  const products = await api("/api/products");
  const h1 = document.createElement("h1");
  h1.textContent = "Products";
  const ul = document.createElement("ul");
  ul.setAttribute("aria-label", "product-list");
  for (const p of products) {
    const li = document.createElement("li");
    const span = document.createElement("span");
    span.textContent = `${p.name} — ${money(p.price)} `;
    const btn = document.createElement("button");
    btn.textContent = "Add to cart";
    btn.addEventListener("click", async () => {
      const cart = await api("/api/cart", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ productId: p.id }),
      });
      setCartCount(cart.count);
      toast("Added to cart");
    });
    li.append(span, btn);
    ul.appendChild(li);
  }
  app.replaceChildren(h1, ul);
}

async function renderCart() {
  app.innerHTML = "<p>Loading cart…</p>";
  const cart = await api("/api/cart");
  const h1 = document.createElement("h1");
  h1.textContent = "Your Cart";
  const table = document.createElement("table");
  table.setAttribute("aria-label", "cart-items");
  table.innerHTML = "<thead><tr><th>Item</th><th>Qty</th><th>Total</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const line of cart.lines) {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = line.name;
    const qty = document.createElement("td");
    qty.textContent = String(line.qty);
    const total = document.createElement("td");
    total.setAttribute("data-testid", "line-total");
    total.textContent = money(line.price * line.qty);
    tr.append(name, qty, total);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  app.replaceChildren(h1, table);
  setCartCount(cart.count);
}

async function renderAccount() {
  app.innerHTML = "<p>Loading account…</p>";
  const h1 = document.createElement("h1");
  h1.textContent = "Account";
  const p = document.createElement("p");
  try {
    const me = await api("/api/me");
    p.textContent = `Signed in as ${me.email}`;
  } catch {
    p.textContent = "Not signed in";
  }
  const today = document.createElement("p");
  today.setAttribute("data-testid", "today");
  today.textContent = `Today is ${new Date().toISOString().slice(0, 10)}`;
  app.replaceChildren(h1, p, today);
}

function renderBroken() {
  const h1 = document.createElement("h1");
  h1.textContent = "Broken";
  app.replaceChildren(h1);
  // a real app bug: throws after render
  setTimeout(() => {
    throw new Error("kaboom: this page has a bug");
  }, 0);
}

async function render() {
  const path = location.pathname;
  if (path === "/cart") return renderCart();
  if (path === "/account") return renderAccount();
  if (path === "/broken") return renderBroken();
  return renderProducts();
}

document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-nav]");
  if (!a) return;
  e.preventDefault();
  history.pushState({}, "", a.getAttribute("href"));
  render();
});

window.addEventListener("popstate", render);
render();
