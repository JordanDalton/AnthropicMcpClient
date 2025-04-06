import { Anthropic } from "@anthropic-ai/sdk";
import {
    MessageParam,
    Tool,
    TextBlock,
    ToolUseBlock,
    ToolResultBlockParam, // Import ToolResultBlockParam
} from "@anthropic-ai/sdk/resources/messages/messages.mjs";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";
import dotenv from "dotenv";
import fs from "fs";
import path from 'path';

dotenv.config();

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY is not set in the environment variables");
}

// --- START: Type Definitions ---

interface ServerParamsStructure {
    command?: string;
    args?: string[];
    env?: Record<string, string>;
}

interface McpServersMap {
    [serverName: string]: ServerParamsStructure;
}

interface ServersFileStructure {
    mcpServers: McpServersMap;
}

interface InternalMPCServerStructure {
    name: string;
    command: string;
    args: string[];
    env?: Record<string, string>;
}


interface AugmentedToolResult {
    content: string | unknown; // Keep original content type for processing
    isError?: boolean;
    tool_use_id: string;
}

// --- END: Type Definitions ---

function filterStringEnv(env: NodeJS.ProcessEnv | Record<string, string | undefined>): Record<string, string> {
    const result: Record<string, string> = {};
    for (const key in env) {
        const value = env[key];
        if (typeof value === 'string') {
            result[key] = value;
        }
    }
    return result;
}


class MCPClient {
    private anthropic: Anthropic;
    private mcpClients: Map<string, Client>;
    private transports: Map<string, StdioClientTransport>;
    private tools: Tool[] = [];
    private toolToServerMap: Map<string, string>;
    private conversationHistory: MessageParam[] = [];
    private systemPrompt?: string;

    constructor(systemPrompt?: string) {
        this.anthropic = new Anthropic({
            apiKey: ANTHROPIC_API_KEY,
        });
        this.mcpClients = new Map();
        this.transports = new Map();
        this.toolToServerMap = new Map();
        this.systemPrompt = systemPrompt;
    }

    async connectToServer(server: InternalMPCServerStructure): Promise<void> {
        console.log(`Attempting connection with structure:`, {
            name: server.name,
            command: server.command,
            args: server.args,
            env: server.env ? '[Custom/Merged Env]' : '[Inherited Env]'
        });
        const serverName = server.name;

        if (this.mcpClients.has(serverName)) {
            console.warn(`Server '${serverName}' is already connected or connection was attempted. Skipping.`);
            return;
        }

        let transport: StdioClientTransport | null = null;
        let mcp: Client | null = null;
        try {
            console.log(`--> Environment for ${serverName}:`, server.env ? 'Custom/Merged' : 'Inherited');

            transport = new StdioClientTransport({
                command: server.command,
                args: server.args,
                env: server.env,
            });
            console.log(`--> Created StdioClientTransport for ${serverName}`);

            mcp = new Client({ name: `mcp-client-cli-${serverName}`, version: "1.0.0" });
            mcp.connect(transport);
            console.log(`--> mcp.connect called for ${serverName}`);

            this.transports.set(serverName, transport);
            this.mcpClients.set(serverName, mcp);

            console.log(`--> Attempting to list tools for ${serverName}...`);
            const listToolsPromise = mcp.listTools();
            const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error(`listTools timed out for ${serverName}`)), 15000));

            const toolsResult = await Promise.race([listToolsPromise, timeoutPromise]) as Awaited<ReturnType<typeof mcp.listTools>>;

            console.log(`--> listTools result for ${serverName}:`, toolsResult);

            const newlyDiscoveredTools: Tool[] = [];
            for (const toolInfo of toolsResult.tools) {
                const tool: Tool = {
                    name: toolInfo.name,
                    description: toolInfo.description,
                    input_schema: toolInfo.inputSchema,
                };

                if (this.toolToServerMap.has(tool.name)) {
                    console.warn(`Tool '${tool.name}' is already provided by server '${this.toolToServerMap.get(tool.name)}'. Server '${serverName}' also provides it. Using the first one discovered.`);
                } else {
                    this.tools.push(tool);
                    this.toolToServerMap.set(tool.name, serverName);
                    newlyDiscoveredTools.push(tool);
                }
            }

            if (newlyDiscoveredTools.length > 0) {
                console.log(
                    `Connected to server '${serverName}' and discovered tools:`,
                    newlyDiscoveredTools.map(({ name }) => name).join(', ')
                );
            } else {
                console.log(`Connected to server '${serverName}'. No new unique tools discovered.`);
            }
            console.log("All available tools:", this.tools.map(t => t.name).join(', ') || 'None');

        } catch (e) {
            console.error(`Failed during connection or tool listing for MCP server '${serverName}': `, e);
            if (mcp && this.mcpClients.get(serverName) === mcp) {
                this.mcpClients.delete(serverName);
            }
            if (transport && this.transports.get(serverName) === transport) {
                this.transports.delete(serverName);
                console.warn(`Removed transport reference for failed server ${serverName}. Underlying process might still be running if startup failed late.`);
            }
            if (mcp) {
                try {
                    console.log(`Attempting to close potentially failed MCP client for ${serverName}...`);
                    await mcp.close();
                } catch (closeError) {
                    console.error(`Error closing MCP client for failed server ${serverName}:`, closeError);
                }
            }
        }
    }

    // --- REFACTORED processQuery ---
    async processQuery(query: string): Promise<string> {
        console.log("\n--- Turn Start ---");
        console.log("Current History Length:", this.conversationHistory.length);

        // 1. Add user message to history
        this.conversationHistory.push({ role: "user", content: query });
        this.ensureHistoryFitsContextWindow();

        const model = "claude-3-5-sonnet-20240620";

        try {
            console.log(`Sending ${this.conversationHistory.length} messages to Anthropic...`);
            const initialApiResponse = await this.anthropic.messages.create({
                model: model,
                max_tokens: 4096,
                system: this.systemPrompt,
                messages: [...this.conversationHistory],
                tools: this.tools.length > 0 ? this.tools : undefined,
            });

            console.log("Anthropic Initial Response Received. Stop Reason:", initialApiResponse.stop_reason);

            // 2. Add assistant's *entire* initial response (including potential tool requests) to history
            this.conversationHistory.push({
                role: 'assistant',
                content: initialApiResponse.content,
            });
            this.ensureHistoryFitsContextWindow(); // Check context again after adding assistant response


            // 3. Extract ALL tool use requests from the response.
            const toolUseBlocks = initialApiResponse.content.filter(
                (block): block is ToolUseBlock => block.type === 'tool_use'
            );

            // 4. Check if the stop reason indicates tool use AND if there are actually tool requests.
            if (initialApiResponse.stop_reason === 'tool_use' && toolUseBlocks.length > 0) {
                console.log(`Assistant requested ${toolUseBlocks.length} tool(s):`, toolUseBlocks.map(t => `${t.name} (ID: ${t.id})`).join(', '));

                const toolResultPromises: Promise<AugmentedToolResult>[] = [];

                // 5. Iterate through EACH tool use request received from Claude.
                for (const toolUse of toolUseBlocks) {
                    const { name: toolName, input: toolInput, id: toolUseId } = toolUse;

                    const serverName = this.toolToServerMap.get(toolName);
                    const targetMcpClient = serverName ? this.mcpClients.get(serverName) : undefined;

                    if (!targetMcpClient) {
                        console.error(`Error: Tool '${toolName}' (ID: ${toolUseId}) requested, but no providing server found or server is not connected.`);
                        // Immediately create an error result for this specific tool request
                        toolResultPromises.push(Promise.resolve({
                            content: `Configuration error: Tool '${toolName}' is not available from any connected server. Please check server status or ensure the tool name is correct.`,
                            isError: true,
                            tool_use_id: toolUseId,
                        }));
                        continue; // Skip to the next tool request in the loop
                    }

                    console.log(`Routing call for tool: ${toolName} (ID: ${toolUseId}) to server: ${serverName}`);

                    // 6. For each tool request, create a promise that represents calling the tool via MCP.
                    //    All these promises are added to the `toolResultPromises` array.
                    toolResultPromises.push(
                        targetMcpClient.callTool({
                            name: toolName,
                            arguments: toolInput as Record<string, unknown> | undefined,
                        })
                        .then((sdkResult): AugmentedToolResult => {
                            // Process successful SDK result
                            console.log(`Raw SDK Result for ${toolName} from ${serverName} (ID: ${toolUseId}):`, JSON.stringify(sdkResult, null, 2));
                            let content: unknown = sdkResult;
                            let isError = false;

                            // Standardize result checking: Look for explicit error or extract content/data
                            if (typeof sdkResult === 'object' && sdkResult !== null) {
                                if ('error' in sdkResult && (sdkResult as any).error) {
                                    isError = true;
                                    content = `Tool Error: ${JSON.stringify((sdkResult as any).error)}`;
                                    console.warn(`Tool ${toolName} (ID: ${toolUseId}) returned an error structure:`, content);
                                } else if ('content' in sdkResult) { // Prefer 'content' if exists
                                    content = (sdkResult as { content: unknown }).content;
                                } else if ('data' in sdkResult) { // Fallback to 'data'
                                    content = (sdkResult as { data: unknown }).data;
                                }
                                // If neither error, content, nor data is present, pass the whole object
                            }
                            // We'll stringify later when creating the ToolResultBlockParam
                            return { content: content, isError: isError, tool_use_id: toolUseId };
                        })
                        .catch((error): AugmentedToolResult => {
                            // Process error during tool call itself (e.g., network issue, MCP error)
                            console.error(`Error calling tool ${toolName} on server ${serverName} (ID: ${toolUseId}):`, error);
                            const clientExists = serverName && this.mcpClients.has(serverName);
                            return {
                                content: `Client-side error executing tool '${toolName}'${clientExists ? '' : ' (server connection may have been lost)'}: ${error instanceof Error ? error.message : String(error)}`,
                                isError: true,
                                tool_use_id: toolUseId,
                            };
                        })
                    );
                } // --- End of loop iterating through tool requests ---

                // 7. Execute ALL the tool call promises concurrently and wait for all to complete.
                const augmentedToolResults: AugmentedToolResult[] = await Promise.all(toolResultPromises);

                // 8. Create ONE user message containing ALL the tool results.
                //    Each result is linked back to the original request via `tool_use_id`.
                const toolResultContents: ToolResultBlockParam[] = augmentedToolResults.map(result => {
                     // Stringify content here, handle potential errors
                     let finalContentString: string;
                     if (typeof result.content === 'string') {
                         finalContentString = result.content;
                     } else {
                         try {
                             finalContentString = JSON.stringify(result.content, null, 2);
                         } catch (stringifyError) {
                             console.error(`Failed to stringify content for tool ${result.tool_use_id}:`, result.content, stringifyError);
                             finalContentString = "[Unstringifiable tool result]";
                             // Consider marking this as an error for Claude if stringification fails
                             result.isError = true;
                         }
                     }

                     // Apply truncation *after* stringification
                     const MAX_RESULT_LENGTH = 8000; // Truncate long results
                     if (finalContentString.length > MAX_RESULT_LENGTH) {
                         console.warn(`Tool result for ${result.tool_use_id} truncated from ${finalContentString.length} to ${MAX_RESULT_LENGTH} chars.`);
                         finalContentString = finalContentString.substring(0, MAX_RESULT_LENGTH) + "... [truncated]";
                     }

                    return {
                        type: 'tool_result',
                        tool_use_id: result.tool_use_id, // Crucial link back to the request
                        content: finalContentString,      // Ensure content is a string for the API
                        is_error: result.isError ?? false, // Ensure is_error is boolean
                    };
                });

                const toolResultParam: MessageParam = {
                    role: 'user', // Results are submitted back in a user role message
                    content: toolResultContents, // Array of ToolResultBlockParam
                };

                // 9. Add this single message with all results to the conversation history.
                this.conversationHistory.push(toolResultParam);
                this.ensureHistoryFitsContextWindow(); // Check context again after adding tool results

                // 10. Call Anthropic again with the updated history (including tool results).
                //     Claude will now generate its final text response based on the outcomes of ALL tool calls.
                console.log("Added tool results to history. Requesting final response from Anthropic...");
                const finalApiResponse = await this.anthropic.messages.create({
                    model: model,
                    max_tokens: 4096,
                    system: this.systemPrompt,
                    messages: [...this.conversationHistory], // Send history including the tool_result message
                    // No 'tools' param needed here, we just want the text response based on tool results
                });

                console.log("Anthropic Final Response Received (after tools). Stop Reason:", finalApiResponse.stop_reason);

                // 11. Add the final assistant text response to history
                this.conversationHistory.push({
                    role: 'assistant',
                    content: finalApiResponse.content,
                });
                 this.ensureHistoryFitsContextWindow(); // Final check

                // 12. Extract and return the final text
                const finalText = finalApiResponse.content
                    .filter((block): block is TextBlock => block.type === 'text')
                    .map(block => block.text)
                    .join('\n');
                console.log("--- Turn End (with tools) ---");
                return finalText || "[Assistant did not provide text after tool use]";

            } else {
                // Handle the case where no tools were called or stop reason wasn't tool_use
                // The initial assistant response is already in history (step 2)
                const initialText = initialApiResponse.content
                    .filter((block): block is TextBlock => block.type === 'text')
                    .map(block => block.text)
                    .join('\n');
                console.log("--- Turn End (no tools required or stop reason wasn't tool_use) ---");
                return initialText || "[Assistant provided no text content]";
            }

        } catch (error: any) {
            console.error("Error during Anthropic API call or processing:", error);
             if (error.name === 'BadRequestError' && error.message?.includes('max_tokens')) {
                 console.error("Error: Anthropic API call failed possibly due to context length or max_tokens setting.", error);
                 // Attempt to remove the last failed assistant message and maybe tool results if they were added
                 if (this.conversationHistory[this.conversationHistory.length - 1]?.role === 'assistant') {
                     this.conversationHistory.pop();
                 }
                 if (this.conversationHistory[this.conversationHistory.length - 1]?.role === 'user' && Array.isArray(this.conversationHistory[this.conversationHistory.length - 1].content) && (this.conversationHistory[this.conversationHistory.length - 1].content as any[])[0]?.type === 'tool_result') {
                     this.conversationHistory.pop();
                 }
                 return "Sorry, the request or response seems to have become too long, potentially after processing tool results. Please try modifying your request or starting a new conversation.";
             } else if (error.name === 'AuthenticationError') {
                 console.error("Error: Anthropic API key is invalid or missing.", error);
                 return "Sorry, there's an issue with the API configuration.";
             } else {
                 console.error("Error during Anthropic API call or processing:", error);
                 return "Sorry, an unexpected error occurred while processing your request with the AI model.";
             }
        }
    } // End of processQuery
    // --- END REFACTORED processQuery ---


    ensureHistoryFitsContextWindow(): void {
        // Very basic history management - consider token counting for more accuracy
        const MAX_HISTORY_MESSAGES = 40; // Keep a reasonable number of turns
        const currentLength = this.conversationHistory.length;

        // Always keep the first message if it's a user message (initial query for context)
        // Or if it's system, although system prompt is separate now.
        const startIndex = (this.conversationHistory[0]?.role === 'user') ? 1 : 0;

        if (currentLength > MAX_HISTORY_MESSAGES) {
            const messagesToRemove = currentLength - MAX_HISTORY_MESSAGES;
            console.warn(`History length (${currentLength}) exceeds max (${MAX_HISTORY_MESSAGES}). Truncating ${messagesToRemove} oldest messages (excluding first).`);
             // Remove from the second message onwards to keep initial context
            this.conversationHistory.splice(startIndex, messagesToRemove);
            console.warn(`History truncated to ${this.conversationHistory.length} messages.`);
        }
    }

    async chatLoop() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        let continueLoop = true;
        rl.on('SIGINT', () => {
            console.log('\nCaught interrupt signal (Ctrl+C). Exiting chat loop...');
            continueLoop = false;
            // Don't close rl here, allow the loop to break naturally
        });

        try {
            console.log("\nMCP Client Started!");
            if (this.systemPrompt) console.log("System Prompt:", this.systemPrompt);
            if (this.mcpClients.size === 0) {
                console.warn("WARNING: No MCP servers were successfully connected and provided tools.");
            }
            if (this.tools.length > 0) {
                console.log("Available tools from all connected servers:", this.tools.map(t => t.name).join(', '));
            } else {
                if (this.mcpClients.size > 0) {
                    console.log("Connected to servers, but no tools were discovered or registered.");
                } else {
                    console.log("No tools available (no servers connected).");
                }
            }
            console.log("Type your queries or 'quit' to exit. Press Ctrl+C to exit gracefully.");

            while (continueLoop) {
                let message = '';
                try {
                     // Use try-catch specifically around rl.question to handle closure during await
                     message = await rl.question("\nYou: ");
                } catch (err: any) {
                     if (err.message === 'closed') { // Check specific error message for readline closure
                          console.log('Readline closed, exiting chat loop.');
                          continueLoop = false;
                     } else {
                          console.error('Readline error:', err); // Log other readline errors
                          continueLoop = false; // Exit on other readline errors too
                     }
                }

                if (!continueLoop) break; // Exit immediately if loop should stop

                const trimmedMessage = message.trim();
                if (trimmedMessage.toLowerCase() === "quit") {
                    continueLoop = false;
                    break;
                }
                if (!trimmedMessage) {
                    continue; // Skip empty input
                }

                console.log("Assistant: Thinking..."); // Provide feedback
                try {
                    const response = await this.processQuery(trimmedMessage);
                    if (!continueLoop) break; // Check again in case SIGINT happened during processQuery
                    console.log("\nAssistant:\n" + response);
                } catch(error) {
                    console.error("\nA critical error occurred in the chat loop's processQuery call:", error);
                    if (!continueLoop) break; // Check again
                    console.log("\nAssistant:\nSorry, a critical internal error occurred. Please check the logs.");
                }
            }
        } finally {
            console.log("Closing readline interface...");
            rl.close();
            console.log("Chat loop finished.");
        }
    }

    async cleanup() {
        console.log("\nInitiating cleanup: Closing MCP connections...");
        const closePromises: Promise<void>[] = [];
        if (this.mcpClients.size > 0) {
            console.log(`Attempting to close ${this.mcpClients.size} MCP client connections...`);
            this.mcpClients.forEach((client, serverName) => {
                console.log(`Closing connection to server '${serverName}'...`);
                // Add a timeout to prevent hangs during close
                const closePromise = client.close()
                    .then(() => console.log(`Successfully closed connection for ${serverName}.`))
                    .catch(err => {
                        console.error(`Error closing MCP connection for server '${serverName}'. Process might be orphaned:`, err);
                    });
                 const timeoutPromise = new Promise<void>((_, reject) => setTimeout(() => reject(new Error(`Timeout closing ${serverName}`)), 5000)); // 5 second timeout
                 closePromises.push(Promise.race([closePromise, timeoutPromise]).catch(err => {
                     console.error(`Timeout or error during close for ${serverName}: ${err.message}`);
                 }));
            });

            await Promise.allSettled(closePromises); // Wait for all close attempts (or timeouts)
            console.log("Finished MCP connection close attempts.");
            this.mcpClients.clear();
            this.transports.clear();
            this.toolToServerMap.clear();
            this.tools = [];
        } else {
            console.log("No active MCP connections to close.");
        }
        // Clear history on cleanup too
        this.conversationHistory = [];
        console.log("Cleanup complete.");
    }
} // End of MCPClient class

// --- Main Function ---

async function main() {
    let serversConfig: ServersFileStructure;
    let mcpClient: MCPClient | null = null;
    let isShuttingDown = false; // Flag to prevent double shutdown

    async function handleShutdown(signal: string) {
        if (isShuttingDown) {
             console.log("Shutdown already in progress...");
             return;
        }
        isShuttingDown = true;
        console.log(`\nCaught ${signal}. Initiating shutdown...`);

        // If chat loop is active, try to interrupt it (readline handler should catch SIGINT)
        // For SIGTERM, we might need to force exit after cleanup attempt

        if (mcpClient) {
            console.log("Running cleanup before exit...");
            try {
                await mcpClient.cleanup();
            } catch(cleanupError) {
                 console.error("Error during cleanup:", cleanupError);
            }
        } else {
            console.log("MCP client not initialized, skipping cleanup.");
        }
        console.log("Exiting process.");
        process.exit(signal === 'SIGINT' ? 0 : 1); // Exit code 0 for SIGINT, 1 for SIGTERM
    }

    // Graceful shutdown setup
    process.on('SIGTERM', () => handleShutdown('SIGTERM'));
    process.on('SIGINT', () => {
        // Let the chat loop's SIGINT handler manage the shutdown if it's running
        if (!mcpClient) { // If client isn't even setup, handle shutdown directly
             handleShutdown('SIGINT');
        } else {
             console.log('\nCaught SIGINT (global handler). Signaling chat loop to exit (or initiating shutdown if loop not active)...');
             // The readline SIGINT handler should set continueLoop = false.
             // If the app is stuck elsewhere, handleShutdown might still be needed,
             // but give the loop a chance first. If it hangs, a second Ctrl+C might be needed.
             // If not in the loop (e.g., during connection), trigger shutdown now.
             if(isShuttingDown) return; // Prevent recursive calls if already shutting down
             // Optional: Add a small delay then check if process is still running, then force shutdown?
        }
    });


    try {
        const serversJsonPath = path.resolve("servers.json");
        console.log(`Reading server configuration from ${serversJsonPath}...`);
        if (!fs.existsSync(serversJsonPath)) {
            throw new Error(`servers.json not found at ${serversJsonPath}`);
        }
        const serversJson = fs.readFileSync(serversJsonPath, "utf-8");
        serversConfig = JSON.parse(serversJson) as ServersFileStructure;

        if (!serversConfig || typeof serversConfig.mcpServers !== 'object' || serversConfig.mcpServers === null) {
            throw new Error("Invalid servers.json structure: 'mcpServers' property is missing or not an object.");
        }
        if (Object.keys(serversConfig.mcpServers).length === 0) {
            console.warn("Warning: 'mcpServers' object in servers.json is empty. No servers to connect to.");
        } else {
            console.log(`Found ${Object.keys(serversConfig.mcpServers).length} server definitions.`);
        }

    } catch (error) {
        console.error("Failed to read or parse servers.json:", error);
        process.exit(1);
    }

    const systemPrompt = " assistant using the Model Context Protocol (MCP) to interact with external tools. When you need to use multiple tools to answer a question, request all of them in a single turn. I will execute them and provide all results back to you. Use the available tools when necessary to fulfill user requests. Be clear and concise in your responses. If a tool fails, report the error clearly based on the information provided.";
    mcpClient = new MCPClient(systemPrompt);

    console.log("\nAttempting to connect to configured servers...");
    const serverEntries = Object.entries(serversConfig.mcpServers);
    const connectionPromises = serverEntries.map(async ([name, serverParams]) => {
        // Add check for shutdown flag during lengthy operations
        if (isShuttingDown) {
            console.log(`Skipping connection for ${name} due to shutdown signal.`);
            return;
        }
        console.log(`\nProcessing server: ${name}`);

        let commandToUse = serverParams?.command;
        if (!commandToUse) {
            console.warn(`Server '${name}' is missing 'command' in servers.json. Skipping.`);
            return;
        }
        if (commandToUse === 'npx' && process.platform === 'win32') {
            commandToUse = 'npx.cmd';
            console.log(`Adjusted command to '${commandToUse}' for Windows platform.`);
        }

        const argsToUse = serverParams?.args ?? [];

        let finalEnv: Record<string, string> | undefined;
        if (serverParams.env) {
            console.log(`Server '${name}' has custom env vars. Merging with process env...`);
            const filteredProcessEnv = filterStringEnv(process.env);
            finalEnv = { ...filteredProcessEnv, ...serverParams.env };
            // Optionally log keys being set/overridden, be careful with sensitive values
            // console.log(`Merged env keys for ${name}: ${Object.keys(finalEnv).join(', ')}`);
        } else {
            finalEnv = filterStringEnv(process.env); // Pass filtered process env even if no custom ones
            // console.log(`Using inherited process env for ${name}.`);
        }


        const mcpServerConfig: InternalMPCServerStructure = {
            name,
            command: commandToUse,
            args: argsToUse,
            env: finalEnv
        };

        if (mcpClient) {
             try {
                 await mcpClient.connectToServer(mcpServerConfig);
             } catch (connectError) {
                  console.error(`Error connecting to server ${name} in main loop:`, connectError)
             }
        } else {
             console.error("MCP Client not initialized when trying to connect (should not happen here).");
        }
    });

    try {
        await Promise.allSettled(connectionPromises);
    } catch (settleError) {
        console.error("Error occurred during Promise.allSettled for connections:", settleError);
    }

    console.log("\nFinished server connection attempts.");

     // Check if shutdown was triggered during connections
     if (isShuttingDown) {
          console.log("Shutdown triggered during server connections. Exiting before chat loop.");
          // Cleanup might have already been called by handleShutdown
          if (mcpClient && !isShuttingDown) await mcpClient.cleanup(); // Avoid double cleanup
          process.exit(0);
     }


    if (mcpClient) {
        console.log("Starting chat loop...");
        try {
            await mcpClient.chatLoop(); // This will run until quit or Ctrl+C
        } catch (error) {
            console.error("A critical error occurred outside the main chat query processing:", error);
             // Ensure cleanup runs even if chatLoop throws unexpectedly
             if (!isShuttingDown) { // Avoid double cleanup if signal handler already ran
                  await handleShutdown('unexpectedError');
             }
        } finally {
             console.log("Chat loop processing finished or exited.");
             // Cleanup should now reliably happen via handleShutdown or natural exit below
        }
    } else {
        console.error("MCP Client instance was not created. Cannot start chat loop.");
    }

    // If the loop finishes normally (e.g., user types 'quit') without a signal
    if (!isShuttingDown) {
        console.log("\nMCP Client application finished normally.");
        if (mcpClient) {
            await mcpClient.cleanup(); // Perform final cleanup
        }
    } else {
        console.log("\nMCP Client application exited due to signal.");
    }

} // End of main function


main().catch(err => {
    console.error("Unhandled error during main execution:", err);
    // Attempt cleanup even on unhandled errors, though it might fail
    const mcpClientInstance = (global as any).mcpClientInstanceForPanicCleanup; // A potential hack if needed, better to rely on handleShutdown
    if (mcpClientInstance && typeof mcpClientInstance.cleanup === 'function') {
         console.log("Attempting emergency cleanup...");
         mcpClientInstance.cleanup().finally(() => process.exit(1));
    } else {
        process.exit(1);
    }
});