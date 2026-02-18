import { z } from "zod";

export const SignupSchema = z.object({
  public_key: z.string().min(1),
  display_name: z.string().min(1).max(100),
});

export const PushRequestSchema = z.object({
  project: z.string().min(1),
  owner: z.string().nullable().optional(),
});

export const PullRequestSchema = z.object({
  project: z.string().min(1),
  since_batch: z.string().nullable().optional(),
  owner: z.string().nullable().optional(),
});

export const ProjectCreateSchema = z.object({
  name: z
    .string()
    .min(1)
    .max(100)
    .regex(/^[a-zA-Z0-9_-]+$/, "project name must be alphanumeric with dashes/underscores"),
});

export const MemberAddSchema = z.object({
  project: z.string().min(1),
  member_account_id: z.string().min(1),
  owner: z.string().nullable().optional(),
});

export const MemberRemoveSchema = z.object({
  project: z.string().min(1),
  member_account_id: z.string().min(1),
  owner: z.string().nullable().optional(),
});

export const MemberListSchema = z.object({
  project: z.string().min(1),
  owner: z.string().nullable().optional(),
});
