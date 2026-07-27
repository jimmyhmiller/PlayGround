import Link from "next/link";
import Head from "next/head";
export default function Home({ time }: { time: string }) {
  return (<main><Head><title>Pages Home</title></Head><h1 id="home">Pages Router Home</h1><p id="time">built {time}</p><Link href="/post/1">Post 1</Link></main>);
}
export async function getStaticProps() { return { props: { time: "static" }, revalidate: 10 }; }
