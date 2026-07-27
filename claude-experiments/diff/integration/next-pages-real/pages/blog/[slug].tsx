import type { GetStaticPaths, GetStaticProps } from "next";
export default function BlogPost({ slug, body }: { slug: string; body: string }) {
  return (<main><h1 id="blog">Blog {slug}</h1><p id="body">{body}</p></main>);
}
export const getStaticPaths: GetStaticPaths = async () => {
  return { paths: [{ params: { slug: "a" } }, { params: { slug: "b" } }], fallback: false };
};
export const getStaticProps: GetStaticProps = async ({ params }) => {
  const slug = String(params!.slug);
  return { props: { slug, body: `static-${slug}` } };
};
