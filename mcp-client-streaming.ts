// mcp-client-streaming.ts
import { Anthropic } from "@anthropic-ai/sdk";
import {
    MessageParam,
    Tool,
    TextBlock,
    ToolUseBlock,
    ToolResultBlockParam,
    MessageStreamEvent, // Keep for potential future use or type checking
    ContentBlock, // Import ContentBlock
} from "@anthropic-ai/sdk/resources/messages/messages.mjs";
import { Client as McpSDKClient } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import dotenv from "dotenv";
import {
    InternalMPCServerStructure,
    // AugmentedToolResult, // Not explicitly used
    SendEventCallback,
    ServersFileStructure
} from "./types.js"; // Assuming types.ts is in the same directory
import { filterStringEnv, truncateString } from "./utils.js"; // Assuming utils.ts is in the same directory
import path from "path";
import fs from "fs"; // Used for logging


dotenv.config();

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY is not set in the environment variables");
}

// --- Log Directory Setup ---
const logDir = path.join("logs");
if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
    console.log(`[Server] Created log directory: ${logDir}`);
}
// --- End Log Directory Setup ---

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
    private processingLock: boolean = false; // Lock to prevent concurrent processing

    constructor(clientId: string, sendEvent: SendEventCallback, systemPrompt?: string) {
        this.clientId = clientId;
        this.anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
        this.mcpClients = new Map();
        this.transports = new Map();
        this.toolToServerMap = new Map();
        this.sendEvent = sendEvent;
        this.systemPrompt = systemPrompt || "You are a helpful assistant.";
        console.log(`[${clientId}] MCPClientStreaming instance created.`);
        console.log(`[${clientId}] System prompt: ${truncateString(this.systemPrompt, 200)}`);
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
            // Windows specific npx path adjustment
            if (commandToUse.toLowerCase() === 'npx' && process.platform === 'win32') {
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

        // Use Promise.allSettled to wait for all connections, even if some fail
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
            return; // Avoid duplicate connections
        }

        let transport: StdioClientTransport | null = null;
        let mcp: McpSDKClient | null = null;
        try {
            transport = new StdioClientTransport({ command: server.command, args: server.args, env: server.env });
            mcp = new McpSDKClient({ name: `mcp-client-sse-${this.clientId}-${serverName}`, version: "1.0.0" });
            mcp.connect(transport);

            this.transports.set(serverName, transport);
            this.mcpClients.set(serverName, mcp); // Add client to map *before* await

            // Set a reasonable timeout for listing tools
            const listToolsPromise = mcp.listTools();
            const timeoutPromise = new Promise((_, reject) =>
                setTimeout(() => reject(new Error(`listTools timed out for ${serverName} after 15s`)), 15000)
            );

            const toolsResult = await Promise.race([listToolsPromise, timeoutPromise]) as Awaited<ReturnType<typeof mcp.listTools>>;

            const newlyDiscoveredTools: Tool[] = [];
            for (const toolInfo of toolsResult.tools) {
                // Explicitly construct the Tool object matching Anthropic's structure
                const tool: Tool = {
                    name: toolInfo.name,
                    description: toolInfo.description,
                    input_schema: toolInfo.inputSchema as Tool['input_schema'], // Cast schema type, assumes compatibility
                };

                // Handle tool name conflicts
                if (this.toolToServerMap.has(tool.name)) {
                    console.warn(`[${this.clientId}] Tool '${tool.name}' conflict: Already provided by '${this.toolToServerMap.get(tool.name)}'. Ignoring from '${serverName}'.`);
                } else {
                    this.tools.push(tool);
                    this.toolToServerMap.set(tool.name, serverName);
                    newlyDiscoveredTools.push(tool); // Add to list for logging/event
                }
            }

            const discoveredNames = newlyDiscoveredTools.map(t => t.name);
            this.sendEvent('tool_connected', { serverName, discoveredTools: discoveredNames });
            console.log(`[${this.clientId}] Connected to ${serverName}. Discovered: ${discoveredNames.join(', ') || 'None'}`);

        } catch (e: any) {
            const errorMsg = e instanceof Error ? e.message : String(e);
            console.error(`[${this.clientId}] Failed during connection/tool listing for ${serverName}: `, e);
            this.sendEvent('tool_connection_failed', { serverName, error: errorMsg });

            // Cleanup partial setup if connection failed
            if (mcp && this.mcpClients.get(serverName) === mcp) this.mcpClients.delete(serverName);
            if (transport && this.transports.get(serverName) === transport) this.transports.delete(serverName);
            if (mcp) {
                try { await mcp.close(); } catch { /* ignore close error during cleanup */ }
            }
            // Do NOT re-throw here, let initializeServers finish other connections
            // throw e; // Re-throwing prevents other servers from connecting
        }
    }

    // --- Main Query Processing (Streaming) ---

    async processQuery(query: string): Promise<void> {
        // Acquire lock
        if (this.processingLock) {
            console.warn(`[${this.clientId}] Processing already in progress. Query ignored: "${query}"`);
            this.sendEvent('error', { message: "Processing already in progress. Please wait for the current turn to complete." });
            return;
        }
        this.processingLock = true;
        this.sendEvent('status', { message: "Processing query..." });
        console.log(`\n[${this.clientId}] --- Turn Start ---`);
        console.log(`[${this.clientId}] Received query: ${query}...`); // Log the received query

        try {
            this.conversationHistory.push({ role: "user", content: query });
            this.ensureHistoryFitsContextWindow(); // Apply truncation *before* API call if needed

            const model = "claude-3-opus-20240229"; //"claude-3-5-sonnet-20240620";
            let currentTurnHistory = [...this.conversationHistory]; // Branch history for this turn

            // *** START OUTER LOOP FOR POTENTIAL MULTI-TURN TOOL USE ***
            let maxToolTurns = 5; // Limit sequential tool calls
            let currentToolTurn = 0;
            let stopReason: string | null = null;

            while (currentToolTurn < maxToolTurns) {
                currentToolTurn++;
                console.log(`[${this.clientId}] Starting Anthropic call (Turn ${currentToolTurn}). History size: ${currentTurnHistory.length}`);
                this.sendEvent('status', { message: `Calling Anthropic (Turn ${currentToolTurn})...` });

                // Log history just before sending
                this.logHistoryState(`BEFORE Anthropic Call (Turn ${currentToolTurn})`, currentTurnHistory);

                // --- Variables specific to THIS Anthropic stream call ---
                let currentAssistantMessageContent: ContentBlock[] = [];
                let currentAssistantMessageText = ""; // Accumulator for plain text within this call
                let currentToolUseBlocks: ToolUseBlock[] = []; // Tool blocks *from this specific call*
                stopReason = null; // Reset stop reason for this call

                const stream = this.anthropic.messages.stream({
                    model: model,
                    max_tokens: 4096,
                    system: this.systemPrompt,
                    messages: currentTurnHistory, // Send the history accumulated *so far in this turn*
                    tools: this.tools.length > 0 ? this.tools : undefined,
                });

                // --- Process Stream ---
                for await (const event of stream) {
                    switch (event.type) {
                        case 'message_start':
                            // Reset accumulators for *this specific* message stream
                            currentAssistantMessageContent = [];
                            currentToolUseBlocks = [];
                            currentAssistantMessageText = "";
                            break;

                        case 'content_block_start':
                            const startBlock = event.content_block;
                            // Initialize block and add to the *current* message content
                            if (startBlock.type === 'text') {
                                currentAssistantMessageContent.push({ type: 'text', text: '', citations: null });
                            } else if (startBlock.type === 'tool_use') {
                                // IMPORTANT: Initialize input as an empty object or string for delta accumulation
                                const newToolBlock: ToolUseBlock = { ...startBlock, input: {} }; // Start with empty object
                                currentAssistantMessageContent.push(newToolBlock);
                                currentToolUseBlocks.push(newToolBlock); // Track separately for easy execution later
                                // Send initial tool_request event (input is empty initially)
                                this.sendEvent('tool_request', { tool_use_id: startBlock.id, name: startBlock.name, input: {} });
                                console.log(`[${this.clientId}] Started tool request: ${startBlock.name} (ID: ${startBlock.id})`);
                            }
                            break;

                        case 'content_block_delta':
                            const delta = event.delta;
                            const blockIndex = event.index;

                            // Ensure the block exists in our accumulator
                            if (!currentAssistantMessageContent[blockIndex]) {
                                console.warn(`[${this.clientId}] Received delta for non-existent block index ${blockIndex}. Ignoring.`);
                                break;
                            }
                            const currentBlock = currentAssistantMessageContent[blockIndex];

                            if (delta.type === 'text_delta' && currentBlock.type === 'text') {
                                this.sendEvent('text_chunk', { text: delta.text });
                                currentAssistantMessageText += delta.text; // Accumulate text for logging/fallback
                                currentBlock.text += delta.text; // Append to the TextBlock in our array
                            } else if (delta.type === 'input_json_delta' && currentBlock.type === 'tool_use') {
                                // *** SAFELY APPEND JSON STRING DELTAS ***
                                // Store partial JSON as a string until the block stops
                                if (typeof currentBlock.input !== 'string') {
                                     // Initialize as string if it's the first delta
                                    currentBlock.input = delta.partial_json;
                                } else {
                                    currentBlock.input += delta.partial_json;
                                }
                            }
                            break;

                        case 'content_block_stop':
                            const stopBlockIndex = event.index;
                            if (!currentAssistantMessageContent[stopBlockIndex]) break; // Should not happen
                            const stoppedBlock = currentAssistantMessageContent[stopBlockIndex];

                            if (stoppedBlock?.type === 'tool_use') {
                                // *** PARSE ACCUMULATED JSON STRING ***
                                let parsedInput: any = {};
                                const inputString = typeof stoppedBlock.input === 'string' ? stoppedBlock.input : ''; // Get accumulated string

                                try {
                                    if (inputString) { // Avoid parsing empty string
                                      parsedInput = JSON.parse(inputString);
                                    } else {
                                      parsedInput = {}; // Default to empty object if no input received
                                    }
                                    stoppedBlock.input = parsedInput; // Update block with PARSED object
                                    // Send updated tool_request event with full parsed input
                                    this.sendEvent('tool_request', { tool_use_id: stoppedBlock.id, name: stoppedBlock.name, input: parsedInput });
                                    console.log(`[${this.clientId}] Completed tool request: ${stoppedBlock.name} (ID: ${stoppedBlock.id}), Input: ${JSON.stringify(parsedInput)}`);
                                } catch (e) {
                                    const errorMsg = `Failed to parse JSON input for tool ${stoppedBlock.id}`;
                                    console.error(`[${this.clientId}] ${errorMsg}:`, inputString, e);
                                    const errorInput = { error: errorMsg, raw: inputString };
                                    stoppedBlock.input = errorInput; // Store error info in the block's input
                                    // Send updated tool_request event showing the error
                                    this.sendEvent('tool_request', { tool_use_id: stoppedBlock.id, name: stoppedBlock.name, input: { error: errorMsg, raw: inputString } });
                                }
                            }
                            break;

                        case 'message_delta':
                            if (event.delta.stop_reason) {
                                stopReason = event.delta.stop_reason;
                                console.log(`[${this.clientId}] Anthropic stream stop reason (Turn ${currentToolTurn}): ${stopReason}`);
                            }
                            break;

                        case 'message_stop':
                            console.log(`[${this.clientId}] Anthropic stream finished (Turn ${currentToolTurn}).`);
                            // Add the completed assistant message from THIS stream call to the TURN'S history
                            const contentToAdd = currentAssistantMessageContent.length > 0
                                ? currentAssistantMessageContent // Contains TextBlock(s) and ToolUseBlock(s) with PARSED/Errored input
                                : [{ type: 'text', text: currentAssistantMessageText || "[No text content]", citations: null } as TextBlock]; // Fallback

                            currentTurnHistory.push({
                                role: 'assistant',
                                content: contentToAdd,
                            });
                            this.ensureHistoryFitsContextWindow(currentTurnHistory); // Truncate the turn's history if needed
                            console.log(`[${this.clientId}] Turn history length after assistant response (Turn ${currentToolTurn}): ${currentTurnHistory.length}`);
                            break;
                    } // End Switch
                } // End stream processing loop for this turn

                // --- Tool Execution (if needed for THIS turn) ---
                if (stopReason === 'tool_use' && currentToolUseBlocks.length > 0) {
                    this.sendEvent('status', { message: `Executing ${currentToolUseBlocks.length} tool(s)...` });
                    console.log(`[${this.clientId}] Assistant requested ${currentToolUseBlocks.length} tool(s) in Turn ${currentToolTurn}:`, currentToolUseBlocks.map(t => `${t.name} (ID: ${t.id})`).join(', '));

                    // Execute tools using the blocks from *this* stream call
                    const toolResultContents = await this.executeTools(currentToolUseBlocks);

                    const toolResultMessage: MessageParam = {
                        role: 'user', // CRITICAL: Role must be 'user' for tool results
                        content: toolResultContents,
                    };

                    // Add tool results to the TURN'S history
                    currentTurnHistory.push(toolResultMessage);
                    this.ensureHistoryFitsContextWindow(currentTurnHistory);
                    console.log(`[${this.clientId}] Turn history length after adding tool results (Turn ${currentToolTurn}): ${currentTurnHistory.length}`);

                    // Loop continues for the next Anthropic call with tool results

                } else {
                    // Stop reason was 'end_turn', 'max_tokens', or no tools were used
                    console.log(`[${this.clientId}] No more tools requested or stop reason is '${stopReason}'. Ending tool loop.`);
                    break; // Exit the while loop
                }

            } // *** END OUTER LOOP FOR POTENTIAL MULTI-TURN TOOL USE ***

            if (currentToolTurn >= maxToolTurns && stopReason === 'tool_use') {
                 console.warn(`[${this.clientId}] Reached maximum tool execution turns (${maxToolTurns}). Stopping interaction.`);
                 this.sendEvent('status', { message: "Reached maximum tool execution turns." });
                 // Optionally add a final message to history indicating this
                 currentTurnHistory.push({ role: 'assistant', content: [{ type: 'text', text: "[Reached maximum tool interaction limit]" }] });
            }

            // --- Finalization ---
            // Update the main conversation history with the final state of the turn's history
            this.conversationHistory = currentTurnHistory;
            console.log(`[${this.clientId}] --- Turn End --- Final History Length: ${this.conversationHistory.length}`);
            this.sendEvent('end', { message: "Processing complete." });

        } catch (error: any) {
            console.error(`[${this.clientId}] Error during processing query:`, error);
            // Adding more context to the error event
            const status = error.status || 'unknown'; // Get status code if available
            let errorContext = `Processing query (status: ${status})`;
            // Refine error context based on common Anthropic errors
            if (status === 400 && error.message?.includes('messages') && error.message?.includes('role')) {
                errorContext = "Likely issue with history structure (e.g., incorrect role sequence)";
                this.logHistoryState(`ERROR STATE (Potential 400 Cause)`, this.conversationHistory); // Log history at time of error
            } else if (status === 400 && error.message?.includes('max_tokens')) {
                errorContext = "Context length potentially exceeded";
            } else if (status === 401 || status === 403) {
                errorContext = "Anthropic authentication/authorization failed";
            } else if (error.name === 'APIConnectionError' || error.name === 'InternalServerError') {
                errorContext = "Anthropic server issue or network problem";
            }

            this.sendEvent('error', {
                message: `An error occurred: ${error.status || 'N/A'} ${error.message || String(error)}`,
                details: error instanceof Error ? { name: error.name, type: (error as any).type ?? 'GenericError', stack: error.stack?.substring(0, 500) } : error,
                context: errorContext
            });
            this.sendEvent('end', { message: "Processing failed." }); // Still send end event
            this.rollbackFailedTurn(); // Attempt to clean up main history
        } finally {
            this.processingLock = false; // Release lock
            this.sendEvent('status', { message: "Ready." }); // Indicate ready state
        }
    }


    private async executeTools(toolUseBlocks: ToolUseBlock[]): Promise<ToolResultBlockParam[]> {
        if (!toolUseBlocks || toolUseBlocks.length === 0) {
            return [];
        }

        const toolResultPromises = toolUseBlocks.map(toolUse => {
            // Safely destructure, provide defaults
            const toolName = toolUse?.name ?? 'unknown_tool';
            const toolInput = toolUse?.input ?? { error: "Missing tool input block" };
            const toolUseId = toolUse?.id ?? `missing_id_${Date.now()}`;

            const serverName = this.toolToServerMap.get(toolName);
            const targetMcpClient = serverName ? this.mcpClients.get(serverName) : undefined;

            // Log the attempt
            console.log(`[${this.clientId}] Preparing to execute tool: ${toolName} (ID: ${toolUseId})`);
            this.logToFile(`${this.clientId}.toolExecuteAttempt.${toolName}.${toolUseId}.log`, toolUse); // Log tool use details

            // --- Input Validation ---
            // Check if input itself contains an error from parsing during the stream
            if (typeof toolInput === 'object' && toolInput !== null && (toolInput as any).error) {
                const errorMsg = `Tool '${toolName}' (ID: ${toolUseId}) called with invalid input: ${(toolInput as any).error}. Raw: ${(toolInput as any).raw || '[N/A]'}`;
                console.error(`[${this.clientId}] ${errorMsg}`);
                this.sendEvent('tool_result', { tool_use_id: toolUseId, name: toolName, content: errorMsg, is_error: true });
                return Promise.resolve({
                    type: 'tool_result',
                    tool_use_id: toolUseId,
                    content: errorMsg,
                    is_error: true,
                } as ToolResultBlockParam);
            }
            // Check if input is missing or not an object/undefined (should be object after parsing)
             if (toolInput === null || typeof toolInput !== 'object') {
                 const errorMsg = `Tool '${toolName}' (ID: ${toolUseId}) called with non-object input: ${JSON.stringify(toolInput)}`;
                 console.error(`[${this.clientId}] ${errorMsg}`);
                 this.sendEvent('tool_result', { tool_use_id: toolUseId, name: toolName, content: errorMsg, is_error: true });
                 return Promise.resolve({
                     type: 'tool_result',
                     tool_use_id: toolUseId,
                     content: errorMsg,
                     is_error: true,
                 } as ToolResultBlockParam);
             }
             // --- End Input Validation ---


            if (!targetMcpClient) {
                const errorMsg = `Configuration error: Tool '${toolName}' (ID: ${toolUseId}) is not available or its server ('${serverName || 'unknown'}') disconnected.`;
                console.error(`[${this.clientId}] ${errorMsg}`);
                this.sendEvent('tool_result', { tool_use_id: toolUseId, name: toolName, content: errorMsg, is_error: true });
                return Promise.resolve({
                    type: 'tool_result',
                    tool_use_id: toolUseId,
                    content: errorMsg,
                    is_error: true,
                } as ToolResultBlockParam);
            }

            // Send 'tool_calling' event before the actual call
            this.sendEvent('tool_calling', { tool_use_id: toolUseId, name: toolName, serverName: serverName!, input: toolInput });
            console.log(`[${this.clientId}] Calling MCP tool: ${toolName} (ID: ${toolUseId}) on server: ${serverName} with input:`, JSON.stringify(toolInput));


            return targetMcpClient.callTool({
                name: toolName,
                 // Ensure arguments is Record<string, unknown> or undefined
                 arguments: (typeof toolInput === 'object' && toolInput !== null && !(toolInput as any).error)
                    ? toolInput as Record<string, unknown>
                    : {}, // Pass empty object if input was invalid/errored earlier
            })
                .then((sdkResult): ToolResultBlockParam => {
                    console.log(`[${this.clientId}] Raw SDK Result for ${toolName} (ID: ${toolUseId}):`, JSON.stringify(sdkResult, null, 2));
                    this.logToFile(`${this.clientId}.toolSdkResult.${toolName}.${toolUseId}.log`, sdkResult); // Log raw SDK result

                    let content: unknown = sdkResult;
                    let isError = false;

                    // Check specifically for MCP error structure first
                    if (typeof sdkResult === 'object' && sdkResult !== null && 'error' in sdkResult && sdkResult.error) {
                        isError = true;
                        const errorDetails = sdkResult.error;
                        content = `Tool Error (${toolName}): ${typeof errorDetails === 'string' ? errorDetails : JSON.stringify(errorDetails)}`;
                        console.warn(`[${this.clientId}] Tool ${toolName} (ID: ${toolUseId}) returned MCP error:`, content);
                    }
                    // Check for common data wrappers if no explicit error
                    else if (typeof sdkResult === 'object' && sdkResult !== null) {
                        if ('content' in sdkResult) content = sdkResult.content;
                        else if ('data' in sdkResult) content = sdkResult.data;
                        // else: the whole object is the content if no error or known wrappers
                    }

                    // Stringify the final content for Anthropic, handle potential circular refs
                    let finalContentString: string;
                    try {
                        finalContentString = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
                    } catch (stringifyError: any) {
                        console.error(`[${this.clientId}] Failed to stringify content for tool ${toolName} (ID: ${toolUseId}):`, content, stringifyError);
                        finalContentString = `[Unstringifiable tool result: ${stringifyError.message}]`;
                        isError = true; // Mark as error if stringification fails
                    }

                    // Truncate potentially large results before sending back
                    const MAX_RESULT_LENGTH = 10000; // Increased limit, adjust as needed
                    const truncatedContent = truncateString(finalContentString, MAX_RESULT_LENGTH);
                    if (finalContentString.length > MAX_RESULT_LENGTH) {
                         console.warn(`[${this.clientId}] Truncated result for tool ${toolName} (ID: ${toolUseId}) from ${finalContentString.length} to ${MAX_RESULT_LENGTH} chars.`);
                    }

                    this.sendEvent('tool_result', { tool_use_id: toolUseId, name: toolName, content: truncatedContent, is_error: isError });
                    return { type: 'tool_result', tool_use_id: toolUseId, content: truncatedContent, is_error: isError };
                })
                .catch((error): ToolResultBlockParam => {
                    // Catch errors during the .callTool() itself (e.g., network issues with MCP server)
                    const errorMsg = `Client-side error calling tool '${toolName}' (ID: ${toolUseId}): ${error instanceof Error ? error.message : String(error)}`;
                    console.error(`[${this.clientId}] ${errorMsg}`, error);
                    this.logToFile(`${this.clientId}.toolClientError.${toolName}.${toolUseId}.log`, { errorMsg, stack: error?.stack });
                    this.sendEvent('tool_result', { tool_use_id: toolUseId, name: toolName, content: errorMsg, is_error: true });
                    return { type: 'tool_result', tool_use_id: toolUseId, content: errorMsg, is_error: true };
                });
        });

        // Wait for all tool calls to complete (successfully or with an error)
        const results = await Promise.all(toolResultPromises);
        console.log(`[${this.clientId}] Finished executing ${toolUseBlocks.length} tool(s).`);
        return results; // Return the array of ToolResultBlockParam
    }


    // --- History Management & Cleanup ---

    // Modified to accept history array as argument for use within processQuery turn
    private ensureHistoryFitsContextWindow(history: MessageParam[] = this.conversationHistory): void {
        const MAX_HISTORY_MESSAGES = 30; // Keep a reasonable limit on message count
        const currentLength = history.length;

        if (currentLength > MAX_HISTORY_MESSAGES) {
            const messagesToRemove = currentLength - MAX_HISTORY_MESSAGES;
            // Always keep the first message (system prompt or initial user query)
            const startIndex = 1;
            if (messagesToRemove > 0 && startIndex < history.length) { // Ensure startIndex is valid
                 const numActuallyRemoved = Math.min(messagesToRemove, history.length - startIndex);
                 if (numActuallyRemoved > 0) {
                     console.warn(`[${this.clientId}] History length (${currentLength}) exceeds max (${MAX_HISTORY_MESSAGES}). Truncating ${numActuallyRemoved} messages from index ${startIndex}.`);
                     history.splice(startIndex, numActuallyRemoved);
                     console.warn(`[${this.clientId}] History truncated to ${history.length} messages.`);
                 }
            }
        }
        // TODO: Add token counting for more precise truncation if needed
    }


    private rollbackFailedTurn(): void {
        // More robust rollback: Remove messages added *during the failed turn*.
        // We assume the turn started with the last 'user' message.
        console.warn(`[${this.clientId}] Attempting to roll back history due to processing error.`);
        if (this.conversationHistory.length > 0) {
            let lastUserIndex = -1;
            for (let i = this.conversationHistory.length - 1; i >= 0; i--) {
                if (this.conversationHistory[i].role === 'user') {
                     // Check if the *next* message is NOT a 'user' message (tool_result is 'user' role)
                     // This helps identify the *actual* start of the user's turn, not a tool result message
                     if (i === this.conversationHistory.length - 1 || this.conversationHistory[i+1]?.role !== 'user') {
                         lastUserIndex = i;
                         break;
                     }
                }
            }

            if (lastUserIndex !== -1) {
                // Remove the user message that started the turn and everything after it
                const removedCount = this.conversationHistory.length - lastUserIndex;
                console.log(`[${this.clientId}] Rolling back ${removedCount} message(s) starting from index ${lastUserIndex} (last user query).`);
                this.conversationHistory.splice(lastUserIndex);
            } else {
                 // Fallback: If only one message, maybe remove it? Or just log.
                 if (this.conversationHistory.length === 1) {
                    console.log(`[${this.clientId}] Rolling back the single initial message.`);
                    this.conversationHistory.pop();
                 } else {
                    console.log(`[${this.clientId}] Rollback heuristic failed: Could not reliably find last user turn start.`);
                 }
            }
            console.log(`[${this.clientId}] History rolled back to length: ${this.conversationHistory.length}`);
            this.logHistoryState("AFTER Rollback", this.conversationHistory);
        }
    }

    // --- Helper Methods ---
    private logHistoryState(context: string, history: MessageParam[]): void {
        console.log(`\n[${this.clientId}] === History State ${context} === (Length: ${history.length})`);
        history.forEach((msg, index) => {
            console.log(`[${this.clientId}] Message ${index}: Role: ${msg.role}`);
            if (typeof msg.content === 'string') {
                console.log(`[${this.clientId}]   Content: "${truncateString(msg.content, 150)}"`);
            } else if (Array.isArray(msg.content)) {
                console.log(`[${this.clientId}]   Content Blocks (${msg.content.length}):`);
                msg.content.forEach((block, blockIndex) => {
                    const blockType = block.type;
                    let details = `Type: ${blockType}`;
                    if (blockType === 'text') details += `, Text: "${truncateString(block.text, 100)}"`;
                    else if (blockType === 'tool_use') details += `, ID: ${block.id}, Name: ${block.name}, Input: ${truncateString(JSON.stringify(block.input), 100)}`;
                    else if (blockType === 'tool_result') details += `, ToolUseID: ${block.tool_use_id}, IsError: ${block.is_error ?? false}, Content: "${truncateString(String(block.content), 100)}"`;
                    console.log(`[${this.clientId}]     Block ${blockIndex}: ${details}`);
                });
            } else {
                console.log(`[${this.clientId}]   Content: [Unknown Format] ${truncateString(JSON.stringify(msg.content), 150)}`);
            }
        });
        console.log(`[${this.clientId}] === End History State ${context} ===\n`);
        // Optionally log to file as well
        this.logToFile(`${this.clientId}.historyState.${context.replace(/[^a-zA-Z0-9]/g, '_')}.log`, history);
    }

    private logToFile(filename: string, data: any): void {
        try {
            const logPath = path.join(logDir, filename);
            const content = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
            fs.writeFileSync(logPath, content + "\n", { flag: 'a' }); // Append mode
        } catch (error) {
            console.error(`[${this.clientId}] Failed to write log file ${filename}:`, error);
        }
    }

    // --- Cleanup ---
    async cleanup(): Promise<void> {
        this.sendEvent('status', { message: "Cleaning up resources..." });
        console.log(`[${this.clientId}] Initiating cleanup...`);
        const closePromises: Promise<void>[] = [];
        this.mcpClients.forEach((client, serverName) => {
            console.log(`[${this.clientId}] Closing connection to server '${serverName}'...`);
            const closePromise = client.close()
                .then(() => console.log(`[${this.clientId}] Successfully closed connection for ${serverName}.`))
                .catch(err => console.error(`[${this.clientId}] Error closing MCP connection for server '${serverName}':`, err));
            // Add a timeout for closing each client
            const timeoutPromise = new Promise<void>((_, reject) =>
                setTimeout(() => reject(new Error(`Timeout closing ${serverName} after 5s`)), 5000)
            );
            closePromises.push(
                Promise.race([closePromise, timeoutPromise])
                .catch(err => { // Catch timeout or close errors
                    console.error(`[${this.clientId}] Timeout or error during close for ${serverName}: ${err.message}`);
                })
            );
        });

        await Promise.allSettled(closePromises); // Wait for all closes/timeouts
        this.mcpClients.clear();
        this.transports.clear(); // Assuming transports are managed alongside clients
        this.toolToServerMap.clear();
        this.tools = [];
        this.conversationHistory = []; // Clear history on cleanup
        this.processingLock = false; // Ensure lock is released
        console.log(`[${this.clientId}] Cleanup complete.`);
        this.sendEvent('status', { message: "Cleanup complete." });
        // Note: We don't close the SSE connection here; the main server route handler does that.
    }

} // End of MCPClientStreaming class