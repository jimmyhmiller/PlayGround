// The default pages-router `App` used when the project has no `pages/_app`.
// Renders the active page with its props, matching Next's built-in App.
export default function App({ Component, pageProps }) {
  return <Component {...pageProps} />;
}
