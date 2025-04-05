// mcp-client-streaming.ts
import { Anthropic } from "@anthropic-ai/sdk";
import {
    MessageParam,
    Tool,
    TextBlock,
    ToolUseBlock,
    ToolResultBlockParam,
    MessageStreamEvent,
} from "@anthropic-ai/sdk/resources/messages/messages.mjs";
import { Client as McpSDKClient } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import dotenv from "dotenv";
import {
    InternalMPCServerStructure,
    AugmentedToolResult,
    SendEventCallback,
    ServersFileStructure
} from "./types.js"; // Assuming types.ts is in the same directory
import { filterStringEnv, truncateString } from "./utils.js"; // Assuming utils.ts is in the same directory
import fs from "fs";
import path from "path";


dotenv.config();

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY is not set in the environment variables");
}

export class MCPClientStreaming {
    private anthropic: Anthropic;
    private mcpClients: Map<string, McpSDKClient>;
    private transports: Map<string, StdioClientTransport>;
    private tools: Tool[] = [];
    private toolToServerMap: Map<string, string>;
    private conversationHistory: MessageParam[] = [];
    private systemPrompt: string;
    private sendEvent: SendEventCallback; // Callback to send SSE events
    private clientId: string; // For context in events

    constructor(clientId: string, sendEvent: SendEventCallback, systemPrompt?: string) {
        this.clientId = clientId;
        this.anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
        this.mcpClients = new Map();
        this.transports = new Map();
        this.toolToServerMap = new Map();
        this.sendEvent = sendEvent;
        this.systemPrompt = systemPrompt || "You are a helpful AI assistant."; // Provide a default
        console.log(`[${clientId}] MCPClientStreaming instance created.`);
    }

    // --- Initialization and Tool Connection ---

    async initializeServers(serversConfig: ServersFileStructure): Promise<void> {
        this.sendEvent('status', { message: "Initializing MCP server connections..." });
        const serverEntries = Object.entries(serversConfig.mcpServers);
        if (serverEntries.length === 0) {
            this.sendEvent('status', { message: "No MCP servers configured." });
            this.sendEvent('all_tools_ready', { availableTools: [] });
            return;
        }

        const connectionPromises = serverEntries.map(async ([name, serverParams]) => {
            this.sendEvent('tool_connecting', { serverName: name });
            console.log(`[${this.clientId}] Processing server: ${name}`);

            let commandToUse = serverParams?.command;
            if (!commandToUse) {
                this.sendEvent('tool_connection_failed', { serverName: name, error: "Missing 'command' in configuration." });
                console.warn(`[${this.clientId}] Server '${name}' is missing 'command'. Skipping.`);
                return;
            }
            if (commandToUse === 'npx' && process.platform === 'win32') {
                commandToUse = 'npx.cmd';
            }

            const argsToUse = serverParams?.args ?? [];
            const finalEnv = serverParams.env
                ? { ...filterStringEnv(process.env), ...serverParams.env }
                : filterStringEnv(process.env);

            const mcpServerConfig: InternalMPCServerStructure = { name, command: commandToUse, args: argsToUse, env: finalEnv };

            try {
                await this.connectToServer(mcpServerConfig);
            } catch (connectError: any) {
                const errorMsg = connectError instanceof Error ? connectError.message : String(connectError);
                this.sendEvent('tool_connection_failed', { serverName: name, error: errorMsg });
                console.error(`[${this.clientId}] Error connecting to server ${name}:`, connectError);
            }
        });

        await Promise.allSettled(connectionPromises);
        const availableTools = this.tools.map(t => t.name);
        this.sendEvent('status', { message: "Finished MCP server connection attempts." });
        this.sendEvent('all_tools_ready', { availableTools });
        console.log(`[${this.clientId}] All server connections processed. Available tools: ${availableTools.join(', ') || 'None'}`);
    }


    private async connectToServer(server: InternalMPCServerStructure): Promise<void> {
        const serverName = server.name;
        console.log(`[${this.clientId}] Attempting connection to ${serverName}...`, server.command, server.args);

        if (this.mcpClients.has(serverName)) {
            console.warn(`[${this.clientId}] Server '${serverName}' already connected or attempted. Skipping.`);
            return;
        }

        let transport: StdioClientTransport | null = null;
        let mcp: McpSDKClient | null = null;
        try {
            transport = new StdioClientTransport({ command: server.command, args: server.args, env: server.env });
            mcp = new McpSDKClient({ name: `mcp-client-sse-${this.clientId}-${serverName}`, version: "1.0.0" });
            mcp.connect(transport);

            this.transports.set(serverName, transport);
            this.mcpClients.set(serverName, mcp);

            const listToolsPromise = mcp.listTools();
            const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error(`listTools timed out for ${serverName}`)), 15000));
            const toolsResult = await Promise.race([listToolsPromise, timeoutPromise]) as Awaited<ReturnType<typeof mcp.listTools>>;

            const newlyDiscoveredTools: Tool[] = [];
            for (const toolInfo of toolsResult.tools) {
                // Ensure we only extract fields defined in the Anthropic Tool type
                const tool: Tool = {
                    name: toolInfo.name,
                    description: toolInfo.description,
                    input_schema: toolInfo.inputSchema as Tool['input_schema'], // Cast schema type
                };

                if (this.toolToServerMap.has(tool.name)) {
                    console.warn(`[${this.clientId}] Tool '${tool.name}' conflict: Already provided by '${this.toolToServerMap.get(tool.name)}'. Ignoring from '${serverName}'.`);
                } else {
                    this.tools.push(tool);
                    this.toolToServerMap.set(tool.name, serverName);
                    newlyDiscoveredTools.push(tool);
                }
            }

            const discoveredNames = newlyDiscoveredTools.map(t => t.name);
            this.sendEvent('tool_connected', { serverName, discoveredTools: discoveredNames });
            console.log(`[${this.clientId}] Connected to ${serverName}. Discovered: ${discoveredNames.join(', ') || 'None'}`);

        } catch (e: any) {
             const errorMsg = e instanceof Error ? e.message : String(e);
            console.error(`[${this.clientId}] Failed during connection/tool listing for ${serverName}: `, e);
             this.sendEvent('tool_connection_failed', { serverName, error: errorMsg });
            // Cleanup partial setup
            if (mcp && this.mcpClients.get(serverName) === mcp) this.mcpClients.delete(serverName);
            if (transport && this.transports.get(serverName) === transport) this.transports.delete(serverName);
            if (mcp) { try { await mcp.close(); } catch { /* ignore close error */ } }
            throw e; // Re-throw so initializeServers knows it failed
        }
    }

    // --- Main Query Processing (Streaming) ---

    // ***** BEGIN UPDATED processQuery method *****
    async processQuery(query: string): Promise<void> {
        this.sendEvent('status', { message: "Processing query..." });
        console.log(`\n[${this.clientId}] --- Turn Start ---`);
        console.log(`[${this.clientId}] History Length: ${this.conversationHistory.length}`);

        this.conversationHistory.push({ role: "user", content: query });
        this.ensureHistoryFitsContextWindow();

        const model = "claude-3-5-sonnet-20240620";
        let fullAssistantResponseContent: (TextBlock | ToolUseBlock)[] = [];
        let requestedTools: ToolUseBlock[] = []; // Still useful for easy access during execution
        let finalStopReason: string | null = null;
        let finalMessageText = ""; // Accumulator specifically for text output

        try {
            this.sendEvent('status', { message: "Calling Anthropic model..." });
            console.log(`[${this.clientId}] Sending ${this.conversationHistory.length} messages to Anthropic (stream)...`);

            const stream = this.anthropic.messages.stream({
                model: model,
                max_tokens: 4096,
                system: this.systemPrompt,
                messages: [...this.conversationHistory],
                tools: this.tools.length > 0 ? this.tools : undefined,
            });

            // --- Start of Corrected Stream Processing ---
            for await (const event of stream) {
                if (event.type === 'message_start') {
                    fullAssistantResponseContent = []; // Reset content array for this message
                    requestedTools = []; // Reset requested tools list
                    finalMessageText = ""; // Reset text accumulator
                } else if (event.type === 'content_block_start') {
                    if (event.content_block.type === 'text') {
                        // Text block is starting, create placeholder in our array
                        const newTextBlock: TextBlock = { type: 'text', text: '', citations: null }; // FIX: Add citations: null
                        fullAssistantResponseContent.push(newTextBlock);
                    } else if (event.content_block.type === 'tool_use') {
                        // Tool use block is starting. event.content_block IS the partial ToolUseBlock here.
                        const currentToolBlock = event.content_block; // Assign directly
                        this.sendEvent('tool_request', { tool_use_id: currentToolBlock.id, name: currentToolBlock.name, input: {} });

                        // Create a *copy* based on the event data for our tracking arrays.
                        // Initialize 'input' specifically for delta accumulation.
                        const newToolBlockForTracking: ToolUseBlock = {
                            id: currentToolBlock.id,
                            name: currentToolBlock.name,
                            input: "", // Initialize input as an empty string for delta accumulation
                            type: 'tool_use'
                        };
                        requestedTools.push(newToolBlockForTracking); // Add the copy for later execution reference
                        fullAssistantResponseContent.push(newToolBlockForTracking); // Add the copy to the main content array being built
                    }
                } else if (event.type === 'content_block_delta') {
                    const delta = event.delta;
                    const blockIndex = event.index; // Index of the block being updated
                    const currentBlock = fullAssistantResponseContent[blockIndex]; // Get the block from our array

                    // Type guard to ensure currentBlock is defined and of the expected type
                    if (delta.type === 'text_delta' && currentBlock?.type === 'text') {
                        this.sendEvent('text_chunk', { text: delta.text });
                        finalMessageText += delta.text; // Accumulate final text separately
                        currentBlock.text += delta.text; // Append to the TextBlock in our array
                    } else if (delta.type === 'input_json_delta' && currentBlock?.type === 'tool_use') {
                        // Append JSON delta string to the input property
                        // Ensure input is treated as a string during accumulation
                        currentBlock.input = (currentBlock.input || "") + delta.partial_json;
                         // Also update the input in the requestedTools tracking array if needed (optional redundancy)
                         const trackedTool = requestedTools.find(t => t.id === currentBlock.id);
                         if(trackedTool) trackedTool.input = currentBlock.input;
                    }
                } else if (event.type === 'content_block_stop') {
                    const blockIndex = event.index;
                    const block = fullAssistantResponseContent[blockIndex];

                    // Type guard
                    if (block?.type === 'tool_use') {
                        // Finalize the input: parse the accumulated JSON string
                        let parsedInput: any = {};
                        try {
                            // Ensure block.input is treated as a string before parsing
                            const inputString = typeof block.input === 'string' ? block.input : JSON.stringify(block.input);
                            parsedInput = JSON.parse(inputString || '{}');
                            block.input = parsedInput; // Update block in main array with parsed object

                            // Update the corresponding tool in requestedTools array as well
                            const trackedTool = requestedTools.find(t => t.id === block.id);
                            if (trackedTool) {
                                trackedTool.input = parsedInput;
                                // Send updated tool_request event with full input now that it's complete
                                this.sendEvent('tool_request', { tool_use_id: block.id, name: block.name, input: parsedInput });
                            } else {
                                 console.warn(`[${this.clientId}] Could not find tool ${block.id} in requestedTools to update parsed input.`);
                            }
                        } catch (e) {
                            console.error(`[${this.clientId}] Failed to parse JSON for tool input ${block.id}:`, block.input, e);
                            const errorInput = { error: "Failed to parse input JSON delta", raw: block.input };
                            block.input = errorInput; // Store error info

                             const trackedTool = requestedTools.find(t => t.id === block.id);
                             if (trackedTool) trackedTool.input = errorInput; // Update tracked tool too

                             // Send update with error state
                             this.sendEvent('tool_request', { tool_use_id: block.id, name: block.name, input: errorInput });
                        }
                    }
                 } else if (event.type === 'message_delta') {
                    if (event.delta.stop_reason) {
                        finalStopReason = event.delta.stop_reason;
                        console.log(`[${this.clientId}] Anthropic stream stop reason: ${finalStopReason}`);
                    }
                 } else if (event.type === 'message_stop') {
                    console.log(`[${this.clientId}] Anthropic message stream finished.`);
                    // Add the complete assistant message to history
                    // Ensure content isn't empty, use accumulated text if needed
                    const contentToAdd = fullAssistantResponseContent.length > 0
                          ? fullAssistantResponseContent
                          : [{ type: 'text', text: finalMessageText || "[No text content]", citations: null } as TextBlock]; // FIX: Add citations
                    this.conversationHistory.push({
                        role: 'assistant',
                        content: contentToAdd,
                    });
                    this.ensureHistoryFitsContextWindow();
                 }
            } // End of stream processing loop
             // --- End of Corrected Stream Processing ---


             // --- Tool Execution (if needed - This part should be mostly okay) ---
             if (finalStopReason === 'tool_use' && requestedTools.length > 0) {
                 this.sendEvent('status', { message: `Executing ${requestedTools.length} tool(s)...` });
                 console.log(`[${this.clientId}] Assistant requested ${requestedTools.length} tool(s):`, requestedTools.map(t => `${t.name} (ID: ${t.id})`).join(', '));

                 // Pass the requestedTools array, which now has parsed input (or error objects)
                 const toolResultContents = await this.executeTools(requestedTools);

                 const toolResultParam: MessageParam = {
                     role: 'user',
                     content: toolResultContents,
                 };
                 this.conversationHistory.push(toolResultParam);
                 this.ensureHistoryFitsContextWindow();

                 // --- Call Anthropic Again with Tool Results ---
                 this.sendEvent('status', { message: "Sending tool results back to Anthropic..." });
                 console.log(`[${this.clientId}] Requesting final response from Anthropic after tools...`);
                 finalMessageText = ""; // Reset final text accumulator for the second response

                 const finalStream = this.anthropic.messages.stream({
                     model: model,
                     max_tokens: 4096,
                     system: this.systemPrompt,
                     messages: [...this.conversationHistory],
                 });

                 // --- Start of Corrected *Final* Response Stream Processing ---
                 let finalAssistantResponseContent: (TextBlock | ToolUseBlock)[] = [];
                 for await (const event of finalStream) {
                      if (event.type === 'content_block_start' && event.content_block.type === 'text') {
                          // Text block starting in the final response
                          const newTextBlock: TextBlock = { type: 'text', text: '', citations: null }; // FIX: Add citations
                          finalAssistantResponseContent.push(newTextBlock);
                      } else if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
                          const blockIndex = event.index;
                          const currentBlock = finalAssistantResponseContent[blockIndex];
                          // Type guard
                          if (currentBlock?.type === 'text') {
                              this.sendEvent('text_chunk', { text: event.delta.text });
                              finalMessageText += event.delta.text; // Accumulate final text
                              currentBlock.text += event.delta.text; // Append to block in array
                          }
                      } else if (event.type === 'message_stop') {
                           console.log(`[${this.clientId}] Anthropic final response stream finished.`);
                           const contentToAdd = finalAssistantResponseContent.length > 0
                               ? finalAssistantResponseContent
                               : [{ type: 'text', text: finalMessageText || "[No text content]", citations: null } as TextBlock]; // FIX: Add citations
                          this.conversationHistory.push({
                              role: 'assistant',
                              content: contentToAdd,
                          });
                          this.ensureHistoryFitsContextWindow();
                      }
                      // Handle other events like message_start, message_delta (stop_reason) if needed
                 }
                 // --- End of Corrected *Final* Response Stream Processing ---

                 console.log(`[${this.clientId}] --- Turn End (with tools) ---`);

             } else {
                 console.log(`[${this.clientId}] --- Turn End (no tools) ---`);
             }

             this.sendEvent('end', { message: "Processing complete." });

        } catch (error: any) {
             console.error(`[${this.clientId}] Error during processing query:`, error);
             let errorContext = "Anthropic API call or stream processing";
              if (error.name === 'BadRequestError' && error.message?.includes('max_tokens')) {
                  errorContext = "Context length potentially exceeded";
              } else if (error.name === 'AuthenticationError') {
                  errorContext = "Anthropic authentication failed";
              }
             this.sendEvent('error', {
                 message: `An error occurred: ${error.message || String(error)}`,
                 details: error instanceof Error ? { name: error.name, stack: error.stack?.substring(0, 500) } : error,
                  context: errorContext
             });
              this.sendEvent('end', { message: "Processing failed." }); // Still send end event
              this.rollbackFailedTurn();
        }
    } // End of processQuery
    // ***** END UPDATED processQuery method *****


    private async executeTools(toolUseBlocks: ToolUseBlock[]): Promise<ToolResultBlockParam[]> {
        const toolResultPromises = toolUseBlocks.map(toolUse => {
            const { name: toolName, input: toolInput, id: toolUseId } = toolUse;
            const serverName = this.toolToServerMap.get(toolName);
            const targetMcpClient = serverName ? this.mcpClients.get(serverName) : undefined;

            // Check if input itself contains an error from parsing
            if (typeof toolInput === 'object' && toolInput !== null && (toolInput as any).error) {
                const errorMsg = `Tool '${toolName}' called with invalid input: ${(toolInput as any).error}. Raw: ${(toolInput as any).raw || '[N/A]'}`;
                console.error(`[${this.clientId}] Error for tool ${toolUseId}: ${errorMsg}`);
                this.sendEvent('tool_result', { tool_use_id: toolUseId, content: errorMsg, is_error: true });
                return Promise.resolve({
                    type: 'tool_result',
                    tool_use_id: toolUseId,
                    content: errorMsg,
                    is_error: true,
                } as ToolResultBlockParam);
            }

            if (!targetMcpClient) {
                const errorMsg = `Configuration error: Tool '${toolName}' is not available or server disconnected.`;
                console.error(`[${this.clientId}] Error for tool ${toolUseId}: ${errorMsg}`);
                 this.sendEvent('tool_result', { tool_use_id: toolUseId, content: errorMsg, is_error: true }); // Send error event immediately
                return Promise.resolve({
                    type: 'tool_result',
                    tool_use_id: toolUseId,
                    content: errorMsg,
                    is_error: true,
                } as ToolResultBlockParam); // Return the block for the array
            }

            this.sendEvent('tool_calling', { tool_use_id: toolUseId, name: toolName, serverName });
            console.log(`[${this.clientId}] Routing call for tool: ${toolName} (ID: ${toolUseId}) to server: ${serverName}`);

            return targetMcpClient.callTool({
                name: toolName,
                arguments: toolInput as Record<string, unknown> | undefined, // Assume valid input structure here
            })
            .then((sdkResult): ToolResultBlockParam => {
                console.log(`[${this.clientId}] Raw SDK Result for ${toolName} (ID: ${toolUseId}):`, JSON.stringify(sdkResult, null, 2));
                let content: unknown = sdkResult;
                let isError = false;

                if (typeof sdkResult === 'object' && sdkResult !== null) {
                    if ('error' in sdkResult && (sdkResult as any).error) {
                        isError = true;
                        content = `Tool Error: ${JSON.stringify((sdkResult as any).error)}`;
                        console.warn(`[${this.clientId}] Tool ${toolName} (ID: ${toolUseId}) returned error:`, content);
                    } else if ('content' in sdkResult) content = (sdkResult as { content: unknown }).content;
                    else if ('data' in sdkResult) content = (sdkResult as { data: unknown }).data;
                }

                let finalContentString: string;
                try {
                    finalContentString = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
                } catch (stringifyError) {
                    console.error(`[${this.clientId}] Failed to stringify content for tool ${toolUseId}:`, content, stringifyError);
                    finalContentString = "[Unstringifiable tool result]";
                    isError = true;
                }

                const MAX_RESULT_LENGTH = 5000; // Truncate long results
                finalContentString = truncateString(finalContentString, MAX_RESULT_LENGTH)

                this.sendEvent('tool_result', { tool_use_id: toolUseId, content: finalContentString, is_error: isError });
                return { type: 'tool_result', tool_use_id: toolUseId, content: finalContentString, is_error: isError };
            })
            .catch((error): ToolResultBlockParam => {
                 const errorMsg = `Client-side error executing tool '${toolName}': ${error instanceof Error ? error.message : String(error)}`;
                console.error(`[${this.clientId}] Error calling tool ${toolName} (ID: ${toolUseId}):`, error);
                 this.sendEvent('tool_result', { tool_use_id: toolUseId, content: errorMsg, is_error: true });
                return { type: 'tool_result', tool_use_id: toolUseId, content: errorMsg, is_error: true };
            });
        });

        // Wait for all tool calls to complete (successfully or with an error)
        const results = await Promise.all(toolResultPromises);
         return results; // Return the array of ToolResultBlockParam
    }


    // --- History Management & Cleanup ---

    private ensureHistoryFitsContextWindow(): void {
        const MAX_HISTORY_MESSAGES = 30; // Reduced max messages for safety
        const currentLength = this.conversationHistory.length;

        if (currentLength > MAX_HISTORY_MESSAGES) {
            const messagesToRemove = currentLength - MAX_HISTORY_MESSAGES;
            // Keep system prompt (if added as first message - though separate now) and remove oldest turns
            // Remove messages in pairs (user + assistant) after the first message if possible
            const startIndex = 1; // Assume first message (initial user query) is important
             if (startIndex < this.conversationHistory.length - messagesToRemove) {
                 console.warn(`[${this.clientId}] History length (${currentLength}) exceeds max (${MAX_HISTORY_MESSAGES}). Truncating ${messagesToRemove} messages.`);
                 this.conversationHistory.splice(startIndex, messagesToRemove);
                 console.warn(`[${this.clientId}] History truncated to ${this.conversationHistory.length} messages.`);
                 // Optional: Send history update event
                 // this.sendEvent('history_update', { history: this.conversationHistory });
             }
        }
    }

    private rollbackFailedTurn(): void {
        // Basic rollback: Remove the last user message and any preceding assistant message from the failed turn
        console.warn(`[${this.clientId}] Rolling back history due to processing error.`);
        if (this.conversationHistory.length > 0) {
            const lastMessage = this.conversationHistory[this.conversationHistory.length - 1];
            // Check if the last message is the user input that caused the error
            if (lastMessage?.role === 'user') {
                this.conversationHistory.pop(); // Remove user message
                 // Check if the one before that was an assistant response (possibly from tool result step)
                 if (this.conversationHistory.length > 0 && this.conversationHistory[this.conversationHistory.length - 1]?.role === 'assistant') {
                      this.conversationHistory.pop(); // Remove assistant message
                 }
            } else if (lastMessage?.role === 'assistant') {
                 // If the error happened after adding an assistant message
                 this.conversationHistory.pop();
                 // Remove the preceding user message and potentially tool results that led to this assistant message
                 // Look backwards for the last 'user' role message
                 let userMessageIndex = -1;
                 for (let i = this.conversationHistory.length - 1; i >= 0; i--) {
                    if (this.conversationHistory[i].role === 'user') {
                        userMessageIndex = i;
                        break;
                    }
                 }
                 // If a user message was found, remove it and anything between it and the popped assistant message
                 // This handles the case where tool results (also user role) might be between the original query and the failed response
                 if (userMessageIndex !== -1) {
                      this.conversationHistory.splice(userMessageIndex);
                 }
            }
             console.log(`[${this.clientId}] History rolled back to length: ${this.conversationHistory.length}`);
             // Optional: Send history update event
             // this.sendEvent('history_update', { history: this.conversationHistory });
        }
    }


    async cleanup(): Promise<void> {
        this.sendEvent('status', { message: "Cleaning up resources..." });
        console.log(`[${this.clientId}] Initiating cleanup...`);
        const closePromises: Promise<void>[] = [];
        this.mcpClients.forEach((client, serverName) => {
            console.log(`[${this.clientId}] Closing connection to server '${serverName}'...`);
            const closePromise = client.close()
                .then(() => console.log(`[${this.clientId}] Successfully closed connection for ${serverName}.`))
                .catch(err => console.error(`[${this.clientId}] Error closing MCP connection for server '${serverName}':`, err));
            const timeoutPromise = new Promise<void>((_, reject) => setTimeout(() => reject(new Error(`Timeout closing ${serverName}`)), 5000));
            closePromises.push(Promise.race([closePromise, timeoutPromise]).catch(err => {
                console.error(`[${this.clientId}] Timeout or error during close for ${serverName}: ${err.message}`);
            }));
        });

        await Promise.allSettled(closePromises);
        this.mcpClients.clear();
        this.transports.clear();
        this.toolToServerMap.clear();
        this.tools = [];
        this.conversationHistory = []; // Clear history on cleanup
        console.log(`[${this.clientId}] Cleanup complete.`);
        this.sendEvent('status', { message: "Cleanup complete." });
        // Note: We don't close the SSE connection here, the server.ts handles that.
    }
} // End of MCPClientStreaming class