// types.ts
import { Tool } from "@anthropic-ai/sdk/resources/messages/messages.mjs";
import { MessageParam, ToolResultBlockParam } from "@anthropic-ai/sdk/resources/messages/messages.mjs";

export interface ServerParamsStructure {
    command?: string;
    args?: string[];
    env?: Record<string, string>;
}

export interface McpServersMap {
    [serverName: string]: ServerParamsStructure;
}

export interface ServersFileStructure {
    mcpServers: McpServersMap;
}

export interface InternalMPCServerStructure {
    name: string;
    command: string;
    args: string[];
    env?: Record<string, string>;
}

export interface AugmentedToolResult {
    content: string | unknown; // Keep original content type for processing
    isError?: boolean;
    tool_use_id: string;
}

// Type for the event callback function
export type SendEventCallback = (event: string, data: any) => void;

// Type definition for SSE Events
export type SseEvent =
    | { type: 'clientId', data: { clientId: string } }
    | { type: 'status', data: { message: string, details?: any } }
    | { type: 'tool_connecting', data: { serverName: string } }
    | { type: 'tool_connected', data: { serverName: string, discoveredTools: string[] } }
    | { type: 'tool_connection_failed', data: { serverName: string, error: string } }
    | { type: 'all_tools_ready', data: { availableTools: string[] } }
    | { type: 'tool_request', data: { tool_use_id: string, name: string, input: any } }
    | { type: 'tool_calling', data: { tool_use_id: string, name: string, serverName: string } }
    | { type: 'tool_result', data: { tool_use_id: string, content: string, is_error: boolean } }
    | { type: 'text_chunk', data: { text: string } }
    | { type: 'final_message', data: { text: string } } // Could be redundant if text_chunk is always used till end
    | { type: 'error', data: { message: string, details?: any, context?: string } }
    | { type: 'history_update', data: { history: MessageParam[] } } // Optional: For debugging/state sync
    | { type: 'end', data: { message?: string } }
    | { type: 'pong', data: { timestamp: number } } // For keep-alive

// Ensure MessageParam is usable if needed elsewhere
export { MessageParam, Tool, ToolResultBlockParam };