// A dynamic `[slug]` route WITH `generateStaticParams` over the LOCAL posts array and
// NO request-scoped read → classified `ssg`: the adapter enumerates the concrete
// params at build and prerenders one static page per post. `dynamicParams` defaults
// true (unlisted slugs render on demand). Also exports route metadata.
import { POSTS, findPost } from "../posts";

export function generateStaticParams() {
  return POSTS.map((p) => ({ slug: p.slug }));
}

export const metadata = {
  title: "blog post",
  description: "a statically-generated blog post",
};

export default async function Post({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const post = findPost(slug);
  return (
    <main id="post">
      <h1>post: {slug}</h1>
      <p>{post ? post.body : "unknown post"}</p>
    </main>
  );
}
