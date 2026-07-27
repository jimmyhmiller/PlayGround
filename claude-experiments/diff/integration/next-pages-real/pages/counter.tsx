import type { GetStaticProps } from "next";
// ISR page: getStaticProps captures a build-time nonce and revalidates every second.
// The served value is stable inside the window (proving static caching) and changes
// once the window elapses (proving regeneration).
export default function Counter({ at }: { at: number }) {
  return (<main><h1 id="counter">Counter</h1><p id="at">{at}</p></main>);
}
export const getStaticProps: GetStaticProps = async () => {
  return { props: { at: Date.now() }, revalidate: 1 };
};
