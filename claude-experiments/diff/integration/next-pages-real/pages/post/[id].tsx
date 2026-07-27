import { useRouter } from "next/router";
export default function Post({ id, ssr }: { id: string; ssr: string }) {
  const r = useRouter();
  return (<main><h1 id="post">Post {id}</h1><p id="ssr">{ssr}</p><button id="back" onClick={() => r.back()}>back</button></main>);
}
export async function getServerSideProps({ params }: { params: { id: string } }) {
  return { props: { id: params.id, ssr: `rendered-per-request-${params.id}` } };
}
