// Standard NextAuth HTTP mount (the course app only used server actions).
// Needed so sessions can be minted programmatically (bat's world session()).
import { handlers } from '@/auth';

export const { GET, POST } = handlers;
