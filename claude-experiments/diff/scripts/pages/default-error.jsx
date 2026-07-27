// The default pages-router error / 404 page used when the project has no
// `pages/_error` and no `pages/404`. Matches Next's minimal built-in error view.
export default function Error({ statusCode, title }) {
  const message =
    title || (statusCode === 404 ? "This page could not be found" : "An error occurred");
  return (
    <div
      id="__diffpack_error"
      style={{ fontFamily: "system-ui, sans-serif", padding: "48px", textAlign: "center" }}
    >
      <h1 style={{ fontSize: "24px", fontWeight: 500 }}>
        {statusCode}
        {" "}
        <span style={{ fontSize: "18px", fontWeight: 300 }}>{message}</span>
      </h1>
    </div>
  );
}
