import type { AppProps } from "next/app";
export default function App({ Component, pageProps }: AppProps) {
  return <div id="app-shell"><Component {...pageProps} /></div>;
}
