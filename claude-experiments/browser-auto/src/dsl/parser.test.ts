import { describe, expect, it } from "vitest";
import { parseFlow, FlowParseError } from "./parser.js";
import { formatAction, formatEffect } from "./ir.js";

const GOOD = `
flow "add item to cart"

given seed "catalog-basic"
given seed "discounts"
given patch products "blue-widget" stock 0
given user "shopper" signed-in
given clock 2026-07-22T10:00:00Z
given stub GET /api/recommendations 200 json []

go /products
  expect heading "Products"
  expect count listitem 12 in list "product-list"

click button "Add to cart" in listitem "Blue Widget"
  expect request POST /api/cart ok
  expect text "1" in testid "cart-count"

click link "Cart"
  expect url /cart
  expect row "Blue Widget" in table "cart-items"
  let price = text in testid "line-total" of row "Blue Widget" in table "cart-items"

click button "Checkout"
  expect url /checkout
  expect text $price in region "order-summary"
`;

describe("parseFlow", () => {
  it("parses the canonical example", () => {
    const flow = parseFlow(GOOD, "good.flow");
    expect(flow.name).toBe("add item to cart");
    expect(flow.givens).toHaveLength(6);
    expect(flow.steps).toHaveLength(4);
    expect(flow.steps[0]!.effects).toHaveLength(2);
    expect(flow.givens[2]).toEqual({ type: "patch", entity: "products", key: "blue-widget", field: "stock", value: 0 });
  });

  it("scopes chained targets right-to-left", () => {
    const flow = parseFlow(GOOD, "good.flow");
    const letEff = flow.steps[2]!.effects.find((e) => e.type === "let");
    expect(letEff).toEqual({
      type: "let",
      name: "price",
      from: {
        kind: "testid",
        name: "line-total",
        within: { kind: "row", name: "Blue Widget", within: { kind: "table", name: "cart-items" } },
      },
    });
  });

  it("parses request effects with methods and status", () => {
    const flow = parseFlow(GOOD, "good.flow");
    expect(flow.steps[1]!.effects[0]).toEqual({ type: "request", method: "POST", pathPattern: "/api/cart", status: "ok" });
  });

  it("round-trips actions/effects through formatters", () => {
    const flow = parseFlow(GOOD, "good.flow");
    expect(formatAction(flow.steps[1]!.action)).toBe('click button "Add to cart" in listitem "Blue Widget"');
    expect(formatEffect(flow.steps[1]!.effects[1]!)).toBe('expect text "1" in testid "cart-count"');
  });

  it("has no way to express a wait — 'wait' is not a token in the grammar", () => {
    expect(() => parseFlow(`flow "t"\nwait 500\n`, "t.flow")).toThrowError(/unknown|expected/);
    expect(() => parseFlow(`flow "t"\ngo /a\n  expect heading "A"\n  wait 500\n`, "t.flow")).toThrowError(
      /indented lines must start with 'expect' or 'let'/,
    );
  });

  it("rejects an action with no effects, explaining why", () => {
    expect(() => parseFlow(`flow "t"\ngo /a\n  expect heading "A"\nclick button "Go"\n`, "t.flow")).toThrowError(
      /declares no effects.*races hide/s,
    );
  });

  it("rejects unknown target kinds with the known list", () => {
    expect(() => parseFlow(`flow "t"\nclick sprocket "X"\n  expect url /x\n`, "t.flow")).toThrowError(
      /unknown target kind "sprocket"/,
    );
  });

  it("rejects $var used before let", () => {
    expect(() => parseFlow(`flow "t"\ngo /a/$id\n  expect heading "A"\n`, "t.flow")).toThrowError(
      /\$id is used before any 'let id/,
    );
  });

  it("rejects givens after the first action", () => {
    expect(() => parseFlow(`flow "t"\ngo /a\n  expect heading "A"\ngiven seed "x"\n`, "t.flow")).toThrowError(
      /'given' must come before the first action/,
    );
  });

  it("collects ALL problems in one pass", () => {
    try {
      parseFlow(`flow "t"\nclick wat "X"\nclick button "A"\nzorp /b\n`, "t.flow");
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(FlowParseError);
      expect((e as FlowParseError).problems.length).toBeGreaterThanOrEqual(3);
    }
  });

  it("rejects bad clock instants", () => {
    expect(() => parseFlow(`flow "t"\ngiven clock not-a-date\ngo /a\n  expect heading "A"\n`, "t.flow")).toThrowError(
      /not a valid ISO-8601/,
    );
  });

  it("parses press with and without scope", () => {
    const flow = parseFlow(
      `flow "t"\npress "Escape"\n  expect no dialog\npress "Enter" in textbox "Search"\n  expect url /results\n`,
      "t.flow",
    );
    expect(flow.steps[0]!.action).toEqual({ type: "press", key: "Escape" });
    expect(flow.steps[0]!.effects[0]).toEqual({ type: "absent", target: { kind: "dialog" } });
    expect(flow.steps[1]!.action).toEqual({ type: "press", key: "Enter", target: { kind: "textbox", name: "Search" } });
  });

  it("parses state effects: checked/unchecked/enabled/disabled/selected", () => {
    const flow = parseFlow(
      `flow "t"
go /f
  expect heading "F"
check field "Paid"
  expect checked field "Paid"
  expect unchecked field "Pending"
  expect enabled button "Submit"
  expect disabled button "Reset"
  expect selected "Lee" in field "Customer"
`,
      "t.flow",
    );
    const effects = flow.steps[1]!.effects;
    expect(effects.map((e) => e.type)).toEqual(["checked", "unchecked", "enabled", "disabled", "selected"]);
    expect(effects[4]).toEqual({ type: "selected", value: "Lee", target: { kind: "field", name: "Customer" } });
  });

  it("parses request body matching and ws frame effects", () => {
    const flow = parseFlow(
      `flow "t"
click button "Save"
  expect request POST /graphql ok containing "mutation CreateInvoice"
  expect request POST /api/x containing "field"
  expect ws sent "hello" on /ws/chat
  expect ws received "echo"
`,
      "t.flow",
    );
    expect(flow.steps[0]!.effects).toEqual([
      { type: "request", method: "POST", pathPattern: "/graphql", status: "ok", bodyContains: "mutation CreateInvoice" },
      { type: "request", method: "POST", pathPattern: "/api/x", status: "ok", bodyContains: "field" },
      { type: "ws", dir: "sent", text: "hello", pathPattern: "/ws/chat" },
      { type: "ws", dir: "received", text: "echo" },
    ]);
  });

  it("parses allow dialogs", () => {
    const flow = parseFlow(`flow "t"\nallow dialogs\ngo /\n  expect heading "H"\n`, "t.flow");
    expect(flow.givens).toEqual([{ type: "allow", what: "dialogs" }]);
  });

  it("parses tab, dialog, download, drag, and frame scope", () => {
    const flow = parseFlow(
      `flow "t"
click link "Open"
  expect tab /terms**
switch tab /terms
  expect heading "Terms"
close tab
  expect heading "Home"
click button "Delete"
  expect dialog "Sure?" accept
  expect download "report.csv"
drag listitem "A" to text "Drop"
  expect text "dropped"
click button "Pay" in frame "checkout"
  expect text "paid" in testid "s" in frame "checkout"
`,
      "t.flow",
    );
    expect(flow.steps[0]!.effects[0]).toEqual({ type: "tab", path: "/terms**" });
    expect(flow.steps[1]!.action).toEqual({ type: "switchTab", path: "/terms" });
    expect(flow.steps[2]!.action).toEqual({ type: "closeTab" });
    expect(flow.steps[3]!.effects).toEqual([
      { type: "dialog", message: "Sure?", response: "accept" },
      { type: "download", name: "report.csv" },
    ]);
    expect(flow.steps[4]!.action).toEqual({
      type: "drag",
      target: { kind: "listitem", name: "A" },
      to: { kind: "text", name: "Drop" },
    });
    expect(flow.steps[5]!.action).toEqual({
      type: "click",
      target: { kind: "button", name: "Pay", within: { kind: "frame", name: "checkout" } },
    });
  });

  it("parses dialog accept with prompt text", () => {
    const flow = parseFlow(`flow "t"\nclick button "Name"\n  expect dialog "Your name?" accept "bat"\n`, "t.flow");
    expect(flow.steps[0]!.effects[0]).toEqual({ type: "dialog", message: "Your name?", response: "accept", text: "bat" });
  });

  it("rejects frame as a bare target — it is a scope only", () => {
    expect(() => parseFlow(`flow "t"\nclick frame "x"\n  expect url /y\n`, "t.flow")).toThrowError(/"frame" is a scope/);
  });

  it("unrolls a table-driven for loop into flat steps at parse time", () => {
    const flow = parseFlow(
      `flow "loop"
go /shop
  expect heading "Shop"
for $item $price in
  "Blue Widget" "$19.99"
  "Red Widget"  "$24.99"
do
  click button "Add" in listitem "$item"
    expect text "$price" in testid "total"
`,
      "loop.flow",
    );
    // 1 go + 2 iterations x 1 body step = 3 flat steps
    expect(flow.steps).toHaveLength(3);
    expect(flow.steps[1]!.action).toEqual({
      type: "click",
      target: { kind: "button", name: "Add", within: { kind: "listitem", name: "Blue Widget" } },
    });
    expect(flow.steps[1]!.effects[0]).toEqual({ type: "text", value: "$19.99", exact: false, target: { kind: "testid", name: "total" } });
    expect(flow.steps[1]!.iteration).toContain('$item="Blue Widget"');
    expect(flow.steps[2]!.action).toMatchObject({ target: { within: { name: "Red Widget" } } });
    expect(flow.steps[2]!.iteration).toContain("iteration 2/2");
  });

  it("expands multi-step loop bodies per iteration", () => {
    const flow = parseFlow(
      `flow "multi"
for $x in
  "a"
  "b"
do
  go /$x
    expect heading "$x"
  click button "next"
    expect url /$x/done
`,
      "m.flow",
    );
    expect(flow.steps).toHaveLength(4); // 2 iterations x 2 body steps
    expect(flow.steps.map((s) => s.source)).toEqual([
      "go /a",
      'click button "next"',
      "go /b",
      'click button "next"',
    ]);
    expect(flow.steps[1]!.effects[0]).toEqual({ type: "url", path: "/a/done" });
    expect(flow.steps[3]!.effects[0]).toEqual({ type: "url", path: "/b/done" });
  });

  it("errors when a for row's value count does not match the variables", () => {
    expect(() =>
      parseFlow(`flow "t"\nfor $a $b in\n  "only-one"\ndo\n  go /$a\n    expect heading "H"\n`, "t.flow"),
    ).toThrowError(/1 value\(s\) but the loop declares 2/);
  });

  it("errors on a for loop with no 'do'", () => {
    expect(() => parseFlow(`flow "t"\nfor $a in\n  "x"\ngo /x\n  expect heading "H"\n`, "t.flow")).toThrowError(
      /missing its 'do'|needs a 'do'/,
    );
  });

  it("supports nested literal loops (cartesian unroll)", () => {
    const flow = parseFlow(
      `flow "t"
for $a in
  "x"
  "y"
do
  for $b in
    "1"
    "2"
  do
    go /$a/$b
      expect heading "$a$b"
`,
      "t.flow",
    );
    expect(flow.steps.map((s) => s.source)).toEqual(["go /x/1", "go /x/2", "go /y/1", "go /y/2"]);
  });

  it("parses a runtime `for each` into a forEach step with a body and $var scope", () => {
    const flow = parseFlow(
      `flow "t"
go /cart
  expect heading "Cart"
for each row in table "cart-items" as $row
  click button "Remove" in $row
    expect gone $row
`,
      "t.flow",
    );
    expect(flow.steps).toHaveLength(2);
    const fe = flow.steps[1]!.action;
    expect(fe.type).toBe("forEach");
    if (fe.type === "forEach") {
      expect(fe.loopVar).toBe("row");
      expect(fe.collection).toEqual({ kind: "row", name: undefined, within: { kind: "table", name: "cart-items" } });
      expect(fe.body).toHaveLength(1);
      expect(fe.body[0]!.action).toEqual({ type: "click", target: { kind: "button", name: "Remove", within: { kind: "ref", name: "row" } } });
      expect(fe.body[0]!.effects[0]).toEqual({ type: "gone", target: { kind: "ref", name: "row" } });
    }
  });

  it("rejects a $scope var that is not an in-scope for-each variable", () => {
    expect(() =>
      parseFlow(`flow "t"\nclick button "x" in $ghost\n  expect url /y\n`, "t.flow"),
    ).toThrowError(/\$ghost is used as a scope but is not a 'for each' loop variable/);
  });

  it("parses fill with $var values and trailing-string extraction", () => {
    const flow = parseFlow(
      `flow "t"\ngo /f\n  expect heading "F"\n  let code = text in testid "code"\nfill textbox "Code" in form "redeem" $code\n  expect value $code in textbox "Code"\n`,
      "t.flow",
    );
    expect(flow.steps[1]!.action).toEqual({
      type: "fill",
      target: { kind: "textbox", name: "Code", within: { kind: "form", name: "redeem" } },
      value: "$code",
    });
  });
});
