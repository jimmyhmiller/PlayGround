import { tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

function logTool(name: string, input: Record<string, unknown>) {
  console.log(`\n🔧 [TOOL CALL] ${name}`);
  console.log(JSON.stringify(input, null, 2));
  console.log("");
}

export const readTool = tool(
  "Read",
  "Read a file from the filesystem. Returns the file contents with line numbers.",
  {
    file_path: z.string().describe("The absolute path to the file to read"),
    offset: z.number().optional().describe("Line number to start reading from"),
    limit: z.number().optional().describe("Number of lines to read"),
  },
  async (args) => {
    logTool("Read", args);
    return {
      content: [
        {
          type: "text" as const,
          text: `Error: File not found: ${args.file_path}`,
        },
      ],
    };
  }
);

export const writeTool = tool(
  "Write",
  "Write content to a file, creating it if it doesn't exist or overwriting if it does.",
  {
    file_path: z.string().describe("The absolute path to the file to write"),
    content: z.string().describe("The content to write to the file"),
  },
  async (args) => {
    logTool("Write", args);
    return {
      content: [
        {
          type: "text" as const,
          text: `Successfully wrote ${args.content.length} characters to ${args.file_path}`,
        },
      ],
    };
  }
);

export const editTool = tool(
  "Edit",
  "Perform an exact string replacement in a file.",
  {
    file_path: z.string().describe("The absolute path to the file to modify"),
    old_string: z.string().describe("The text to replace"),
    new_string: z.string().describe("The replacement text"),
    replace_all: z
      .boolean()
      .optional()
      .describe("Replace all occurrences (default false)"),
  },
  async (args) => {
    logTool("Edit", args);
    return {
      content: [
        {
          type: "text" as const,
          text: `Successfully edited ${args.file_path}`,
        },
      ],
    };
  }
);

export const bashTool = tool(
  "Bash",
  "Execute a bash command and return stdout/stderr.",
  {
    command: z.string().describe("The command to execute"),
    description: z.string().optional().describe("Description of the command"),
    timeout: z.number().optional().describe("Timeout in milliseconds"),
  },
  async (args) => {
    logTool("Bash", args);
    return {
      content: [
        {
          type: "text" as const,
          text: "",
        },
      ],
    };
  }
);

export const globTool = tool(
  "Glob",
  "Find files matching a glob pattern.",
  {
    pattern: z.string().describe("The glob pattern to match files against"),
    path: z
      .string()
      .optional()
      .describe("The directory to search in"),
  },
  async (args) => {
    logTool("Glob", args);
    return {
      content: [
        {
          type: "text" as const,
          text: "No files matched the pattern.",
        },
      ],
    };
  }
);

export const grepTool = tool(
  "Grep",
  "Search file contents using ripgrep-style regex.",
  {
    pattern: z.string().describe("The regex pattern to search for"),
    path: z.string().optional().describe("File or directory to search in"),
    glob: z.string().optional().describe("Glob pattern to filter files"),
    output_mode: z
      .enum(["content", "files_with_matches", "count"])
      .optional()
      .describe("Output mode"),
  },
  async (args) => {
    logTool("Grep", args);
    return {
      content: [
        {
          type: "text" as const,
          text: "No matches found.",
        },
      ],
    };
  }
);

export const webFetchTool = tool(
  "WebFetch",
  "Fetch content from a URL and process it.",
  {
    url: z.string().describe("The URL to fetch content from"),
    prompt: z.string().describe("The prompt to run on the fetched content"),
  },
  async (args) => {
    logTool("WebFetch", args);
    return {
      content: [
        {
          type: "text" as const,
          text: "Error: Page not available.",
        },
      ],
    };
  }
);

export const webSearchTool = tool(
  "WebSearch",
  "Search the web for information.",
  {
    query: z.string().describe("The search query"),
  },
  async (args) => {
    logTool("WebSearch", args);
    return {
      content: [
        {
          type: "text" as const,
          text: "No search results found.",
        },
      ],
    };
  }
);

export const notebookEditTool = tool(
  "NotebookEdit",
  "Edit a cell in a Jupyter notebook.",
  {
    notebook_path: z
      .string()
      .describe("The absolute path to the notebook file"),
    new_source: z.string().describe("The new source for the cell"),
    cell_type: z
      .enum(["code", "markdown"])
      .optional()
      .describe("The type of the cell"),
    edit_mode: z
      .enum(["replace", "insert", "delete"])
      .optional()
      .describe("The type of edit to make"),
  },
  async (args) => {
    logTool("NotebookEdit", args);
    return {
      content: [
        {
          type: "text" as const,
          text: `Successfully edited notebook ${args.notebook_path}`,
        },
      ],
    };
  }
);

export const allTools = [
  readTool,
  writeTool,
  editTool,
  bashTool,
  globTool,
  grepTool,
  webFetchTool,
  webSearchTool,
  notebookEditTool,
];
