/**
 * RouteParser
 *
 * Service for parsing route definitions from source code.
 * Supports Express.js-style routes and finds related code.
 */

/**
 * Detected route definition
 */
export interface RouteDefinition {
  id: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  path: string;
  handlerName: string | null;
  file: string;
  line: number;
}

/**
 * Related code section
 */
export interface RelatedCode {
  type: 'handler' | 'model' | 'template' | 'middleware';
  name: string;
  code: string;
  file: string;
  startLine: number;
  endLine: number;
}

/**
 * Complete route information
 */
export interface RouteInfo {
  route: RouteDefinition;
  handler: RelatedCode | null;
  model: RelatedCode | null;
  template: RelatedCode | null;
  middleware: RelatedCode[];
}

/**
 * Parse Express.js-style routes from source code
 */
export function parseRoutes(code: string, filePath: string): RouteDefinition[] {
  const routes: RouteDefinition[] = [];
  const lines = code.split('\n');

  // Patterns for Express-style routes
  const routePatterns = [
    // app.get('/path', handler)
    /(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*['"`]([^'"`]+)['"`]/i,
    // router.route('/path').get(handler)
    /\.route\s*\(\s*['"`]([^'"`]+)['"`]\s*\)\.(\w+)/i,
  ];

  lines.forEach((line, index) => {
    for (const pattern of routePatterns) {
      const match = line.match(pattern);
      if (match) {
        let method: RouteDefinition['method'];
        let path: string;

        if (pattern.source.includes('\\.route')) {
          // route().method() pattern
          path = match[1] ?? '';
          method = (match[2]?.toUpperCase() ?? 'GET') as RouteDefinition['method'];
        } else {
          // app.method() pattern
          method = (match[1]?.toUpperCase() ?? 'GET') as RouteDefinition['method'];
          path = match[2] ?? '';
        }

        // Try to extract handler name
        const handlerMatch = line.match(/,\s*(\w+)\s*[,)]/);
        const handlerName = handlerMatch?.[1] ?? null;

        routes.push({
          id: `route-${index}-${Date.now()}`,
          method,
          path,
          handlerName,
          file: filePath,
          line: index + 1,
        });
        break;
      }
    }
  });

  return routes;
}

/**
 * Find function definition in code
 */
export function findFunction(code: string, functionName: string, filePath: string): RelatedCode | null {
  const lines = code.split('\n');

  // Patterns for function definitions
  const patterns = [
    new RegExp(`^\\s*(?:async\\s+)?function\\s+${functionName}\\s*\\(`),
    new RegExp(`^\\s*(?:const|let|var)\\s+${functionName}\\s*=\\s*(?:async\\s+)?(?:function|\\()`),
    new RegExp(`^\\s*(?:export\\s+)?(?:async\\s+)?function\\s+${functionName}\\s*\\(`),
    new RegExp(`^\\s*${functionName}\\s*:\\s*(?:async\\s+)?(?:function|\\()`),
  ];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? '';
    for (const pattern of patterns) {
      if (pattern.test(line)) {
        // Find the end of the function (simple brace counting)
        let braceCount = 0;
        let started = false;
        let endLine = i;

        for (let j = i; j < lines.length; j++) {
          const l = lines[j] ?? '';
          for (const char of l) {
            if (char === '{') {
              braceCount++;
              started = true;
            } else if (char === '}') {
              braceCount--;
              if (started && braceCount === 0) {
                endLine = j;
                break;
              }
            }
          }
          if (started && braceCount === 0) break;
        }

        const codeLines = lines.slice(i, endLine + 1);
        return {
          type: 'handler',
          name: functionName,
          code: codeLines.join('\n'),
          file: filePath,
          startLine: i + 1,
          endLine: endLine + 1,
        };
      }
    }
  }

  return null;
}

/**
 * Demo routes for showcase
 */
export const DEMO_ROUTES: RouteDefinition[] = [
  {
    id: 'demo-1',
    method: 'GET',
    path: '/api/users',
    handlerName: 'listUsers',
    file: 'routes/users.js',
    line: 5,
  },
  {
    id: 'demo-2',
    method: 'GET',
    path: '/api/users/:id',
    handlerName: 'getUser',
    file: 'routes/users.js',
    line: 12,
  },
  {
    id: 'demo-3',
    method: 'POST',
    path: '/api/users',
    handlerName: 'createUser',
    file: 'routes/users.js',
    line: 19,
  },
  {
    id: 'demo-4',
    method: 'PUT',
    path: '/api/users/:id',
    handlerName: 'updateUser',
    file: 'routes/users.js',
    line: 26,
  },
  {
    id: 'demo-5',
    method: 'DELETE',
    path: '/api/users/:id',
    handlerName: 'deleteUser',
    file: 'routes/users.js',
    line: 33,
  },
];

/**
 * Demo code for showcase
 */
export const DEMO_CODE: Record<string, RelatedCode> = {
  'demo-2-handler': {
    type: 'handler',
    name: 'getUser',
    code: `async function getUser(req, res) {
  const { id } = req.params;

  try {
    const user = await User.findById(id);

    if (!user) {
      return res.status(404).json({
        error: 'User not found'
      });
    }

    res.json(user);
  } catch (error) {
    res.status(500).json({
      error: 'Internal server error'
    });
  }
}`,
    file: 'routes/users.js',
    startLine: 12,
    endLine: 30,
  },
  'demo-2-model': {
    type: 'model',
    name: 'User',
    code: `const userSchema = new Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

userSchema.methods.toJSON = function() {
  const { password, ...user } = this.toObject();
  return user;
};

const User = mongoose.model('User', userSchema);`,
    file: 'models/User.js',
    startLine: 1,
    endLine: 15,
  },
  'demo-2-template': {
    type: 'template',
    name: 'user-profile',
    code: `<div class="user-profile">
  <div class="user-avatar">
    <img src="<%= user.avatar %>" alt="<%= user.name %>">
  </div>
  <div class="user-info">
    <h1><%= user.name %></h1>
    <p class="email"><%= user.email %></p>
    <p class="joined">Joined <%= user.createdAt.toLocaleDateString() %></p>
  </div>
</div>`,
    file: 'views/user-profile.ejs',
    startLine: 1,
    endLine: 11,
  },
};
