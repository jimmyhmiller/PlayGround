# Known app bugs (flows here are correct; the app is not)

## invoice-delete.flow — delete succeeds in the DB, UI updates only ~half the time

`deleteInvoice` (app/lib/actions.ts) runs `DELETE FROM invoices ...` then
`revalidatePath('/dashboard/invoices')`. The database row is deleted on **every**
click (verified by polling Postgres during `scripts/delete-probe.ts`), but the
row disappears from the page only ~50% of the time — even for a fully hydrated
page with a 1.5s "human" pause before clicking:

```
attempt 1: db invoices=2 (3→2 expected) | DOM row gone: +276ms
attempt 2: db invoices=2 (3→2 expected) | DOM row gone: NEVER (6s)
attempt 3: db invoices=2 (3→2 expected) | DOM row gone: +239ms
attempt 4: db invoices=2 (3→2 expected) | DOM row gone: NEVER (6s)
```

A real user clicks Delete and, half the time, the invoice appears to survive
until they reload. Suspected mechanism: the server action's revalidation
response racing the `<Suspense key={query + currentPage}>`-keyed table segment
in the app router — the flight response resolves against a segment key that
loses the race and gets dropped.

bat's run of this flow reports exactly this shape: `expect request POST
/dashboard/invoices ok` passes, the row-count/absence expectations fail,
reproducibility is "NOT deterministic" with identical network traffic across
runs — pointing the investigation at rendering/app state rather than the
network, which is where the bug turned out to be.
