import { AppError } from "./errors.js";

export class RateLimitError extends AppError {
  constructor(retryAfterMs: number) {
    super(429, "rate limit exceeded");
    this.name = "RateLimitError";
    this.retryAfterMs = retryAfterMs;
  }
  retryAfterMs: number;
}

interface BucketConfig {
  maxRequests: number;
  windowMs: number;
}

// Default limits per category
const DEFAULT_LIMITS: Record<string, BucketConfig> = {
  // Auth: tight limits to prevent brute-force
  auth: { maxRequests: 10, windowMs: 60_000 },
  // Sync: moderate — clients push/pull in bursts
  sync: { maxRequests: 60, windowMs: 60_000 },
  // Projects/members: moderate
  projects: { maxRequests: 30, windowMs: 60_000 },
};

interface Bucket {
  timestamps: number[];
}

export class RateLimiter {
  private buckets = new Map<string, Bucket>();
  private limits: Record<string, BucketConfig>;
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor(limits?: Record<string, BucketConfig>) {
    this.limits = limits ?? DEFAULT_LIMITS;
  }

  /** Categorize a path into a rate-limit bucket category. */
  static categorize(path: string): string {
    const clean = path.split("?")[0];
    if (clean.startsWith("/auth/")) return "auth";
    if (clean.startsWith("/sync/")) return "sync";
    return "projects";
  }

  /**
   * Check rate limit for a given account + category.
   * Throws RateLimitError if exceeded.
   */
  check(accountId: string, category: string): void {
    const config = this.limits[category] ?? this.limits["projects"]!;
    const key = `${accountId}:${category}`;
    const now = Date.now();
    const windowStart = now - config.windowMs;

    let bucket = this.buckets.get(key);
    if (!bucket) {
      bucket = { timestamps: [] };
      this.buckets.set(key, bucket);
    }

    // Prune old entries
    bucket.timestamps = bucket.timestamps.filter((t) => t > windowStart);

    if (bucket.timestamps.length >= config.maxRequests) {
      const oldestInWindow = bucket.timestamps[0];
      const retryAfter = oldestInWindow + config.windowMs - now;
      throw new RateLimitError(Math.max(retryAfter, 1000));
    }

    bucket.timestamps.push(now);
  }

  /** Start periodic cleanup of stale buckets. */
  startCleanup(intervalMs = 60_000): void {
    this.cleanupInterval = setInterval(() => {
      const now = Date.now();
      for (const [key, bucket] of this.buckets) {
        bucket.timestamps = bucket.timestamps.filter(
          (t) => t > now - 120_000,
        );
        if (bucket.timestamps.length === 0) {
          this.buckets.delete(key);
        }
      }
    }, intervalMs);
  }

  stopCleanup(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }

  /** Reset all state (for testing). */
  reset(): void {
    this.buckets.clear();
  }
}
