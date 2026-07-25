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
  setTimeout(() => el.remove(), window.__batTiming?.toastMs ?? 150);
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

async function renderSearch() {
  const h1 = document.createElement("h1");
  h1.textContent = "Search";
  const input = document.createElement("input");
  input.type = "search";
  input.setAttribute("aria-label", "Search");
  const ul = document.createElement("ul");
  ul.setAttribute("aria-label", "search-results");
  const hint = document.createElement("p");
  hint.textContent = "Type to search the catalog.";
  let debounce = null;
  input.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(async () => {
      const products = await api(`/api/products?q=${encodeURIComponent(input.value)}`);
      ul.replaceChildren(
        ...products.map((p) => {
          const li = document.createElement("li");
          li.textContent = `${p.name} — ${money(p.price)}`;
          return li;
        }),
      );
      hint.textContent = `${products.length} result(s)`;
    }, 300);
  });
  app.replaceChildren(h1, input, hint, ul);
}

async function renderFlakyCart() {
  // DELIBERATE BUG (the classic refetch race): the click handler fires the
  // mutation AND a fire-and-forget refetch at the same time. The refetch reads
  // the server's pre-write state; whichever response lands LAST wins the badge.
  app.innerHTML = "<p>Loading…</p>";
  const products = await api("/api/products");
  const p = products[0];
  const h1 = document.createElement("h1");
  h1.textContent = "Flaky Cart";
  const label = document.createElement("p");
  label.textContent = `${p.name} — ${money(p.price)}`;
  const count = document.createElement("span");
  count.setAttribute("data-testid", "flaky-count");
  count.textContent = "0";
  const countWrap = document.createElement("p");
  countWrap.append("in cart: ", count);
  const btn = document.createElement("button");
  btn.textContent = "Add to cart";
  btn.addEventListener("click", () => {
    api("/api/cart", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ productId: p.id }),
    }).then((cart) => {
      count.textContent = String(cart.count);
    });
    // BUG: refetch not sequenced after the mutation
    api("/api/cart").then((cart) => {
      count.textContent = String(cart.count);
    });
  });
  app.replaceChildren(h1, label, countWrap, btn);
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

function renderChat() {
  const h1 = document.createElement("h1");
  h1.textContent = "Chat";
  const input = document.createElement("input");
  input.setAttribute("aria-label", "Message");
  const btn = document.createElement("button");
  btn.textContent = "Send";
  const ul = document.createElement("ul");
  ul.setAttribute("aria-label", "messages");
  const ws = new WebSocket(`ws://${location.host}/ws/chat`);
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    const li = document.createElement("li");
    li.textContent = `${msg.user}: ${msg.text}`;
    ul.appendChild(li);
  });
  btn.addEventListener("click", () => {
    if (!input.value) return;
    const li = document.createElement("li");
    li.textContent = `me: ${input.value}`;
    ul.appendChild(li);
    ws.send(JSON.stringify({ text: input.value }));
    input.value = "";
  });
  app.replaceChildren(h1, input, btn, ul);
}

function renderTicker() {
  const h1 = document.createElement("h1");
  h1.textContent = "Ticker";
  const p = document.createElement("p");
  p.setAttribute("data-testid", "tick");
  p.textContent = "waiting…";
  const es = new EventSource("/api/ticker");
  es.addEventListener("message", (ev) => {
    p.textContent = `tick ${ev.data}`;
  });
  app.replaceChildren(h1, p);
}

async function renderManageCart() {
  // A dynamic list whose size is only known at runtime: rows come from the
  // seeded cart. Each Remove deletes that row (server + DOM), so the count
  // shrinks as you iterate — the case bat's `for each` must handle.
  app.innerHTML = "<h1>Manage Cart</h1><p>items: <span data-testid=\"count\">?</span></p>";
  const table = document.createElement("table");
  table.setAttribute("aria-label", "cart-items");
  const tbody = document.createElement("tbody");
  table.appendChild(tbody);
  app.appendChild(table);
  async function refresh() {
    const cart = await api("/api/cart");
    document.querySelector('[data-testid="count"]').textContent = String(cart.lines.length);
    tbody.replaceChildren(
      ...cart.lines.map((line) => {
        const tr = document.createElement("tr");
        tr.setAttribute("aria-label", line.name);
        const name = document.createElement("td");
        name.textContent = line.name;
        const cell = document.createElement("td");
        const btn = document.createElement("button");
        btn.textContent = "Remove";
        btn.addEventListener("click", async () => {
          await api("/api/cart/remove", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ name: line.name }),
          });
          tr.remove();
          const c = await api("/api/cart");
          document.querySelector('[data-testid="count"]').textContent = String(c.lines.length);
        });
        cell.appendChild(btn);
        tr.append(name, cell);
        return tr;
      }),
    );
  }
  await refresh();
}

function renderInteractions() {
  app.innerHTML = `
    <h1>Interactions</h1>
    <a href="/terms" target="_blank" rel="opener">Open Terms</a>
    <button id="del">Delete account</button>
    <p data-testid="del-status">idle</p>
    <button id="dl">Download report</button>
    <ul aria-label="drag-list">
      <li draggable="true" id="src">Draggable item</li>
    </ul>
    <div id="drop" style="border:1px solid #ccc;padding:1rem">Drop zone: <span data-testid="drop-status">empty</span></div>
    <iframe title="widget" src="/widget" style="width:300px;height:120px"></iframe>
  `;
  document.getElementById("del").addEventListener("click", () => {
    const ok = confirm("Really delete your account?");
    document.querySelector('[data-testid="del-status"]').textContent = ok ? "deleted" : "cancelled";
  });
  document.getElementById("dl").addEventListener("click", () => {
    const blob = new Blob(["item,qty\nBlue Widget,1\n"], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "report.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  const src = document.getElementById("src");
  const drop = document.getElementById("drop");
  src.addEventListener("dragstart", (e) => e.dataTransfer.setData("text", "item"));
  drop.addEventListener("dragover", (e) => e.preventDefault());
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    document.querySelector('[data-testid="drop-status"]').textContent = "dropped";
  });
}

function renderWidget() {
  document.body.innerHTML = '<button id="pay">Pay now</button><p data-testid="widget-status">ready</p>';
  document.getElementById("pay").addEventListener("click", () => {
    document.querySelector('[data-testid="widget-status"]').textContent = "paid";
  });
}

function renderTerms() {
  document.body.innerHTML = "<h1>Terms of Service</h1><p>Be excellent to each other.</p>";
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
  if (path === "/search") return renderSearch();
  if (path === "/flaky-cart") return renderFlakyCart();
  if (path === "/chat") return renderChat();
  if (path === "/ticker") return renderTicker();
  if (path === "/manage-cart") return renderManageCart();
  if (path === "/interactions") return renderInteractions();
  if (path === "/widget") return renderWidget();
  if (path === "/terms") return renderTerms();
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
