// LOCAL data (the hermeticity contract): the blog's posts are a static array — no
// fetch, no network. `generateStaticParams` enumerates these at build time.
export interface Post {
  slug: string;
  title: string;
  body: string;
}

export const POSTS: Post[] = [
  { slug: "hello", title: "Hello", body: "the first post" },
  { slug: "intro", title: "Intro", body: "the second post" },
];

export function findPost(slug: string): Post | undefined {
  return POSTS.find((p) => p.slug === slug);
}
