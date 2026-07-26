import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: "https://example.com",
      lastModified: new Date("2024-01-01T00:00:00.000Z"),
      changeFrequency: "yearly",
      priority: 1,
    },
    {
      url: "https://example.com/blog",
      lastModified: new Date("2024-02-01T00:00:00.000Z"),
      changeFrequency: "weekly",
      priority: 0.5,
    },
  ];
}
