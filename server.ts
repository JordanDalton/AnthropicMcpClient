// server.ts
// **** Import ParamsDictionary from the correct source ****
import express, { Request, Response, NextFunction, RequestHandler } from 'express';
import { ParamsDictionary } from 'express-serve-static-core'; // Correct import location
import { ParsedQs } from 'qs';
import cors from 'cors';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { MCPClientStreaming } from './mcp-client-streaming.js';
import { ServersFileStructure, SendEventCallback, SseEvent } from './types.js';
import { formatSseEvent } from './utils.js';
const clientLocks = new Map<string, boolean>(); // clientId -> 

dotenv.config();

const PORT = process.env.PORT || 3005;
const app = express();

// --- Configuration Loading ---
let serversConfig: ServersFileStructure;
try {
    const serversJsonPath = path.resolve("servers.json");
    console.log(`Reading server configuration from ${serversJsonPath}...`);
    if (!fs.existsSync(serversJsonPath)) {
        throw new Error(`servers.json not found at ${serversJsonPath}`);
    }
    const serversJson = fs.readFileSync(serversJsonPath, "utf-8");
    serversConfig = JSON.parse(serversJson) as ServersFileStructure;
    if (!serversConfig?.mcpServers) {
        throw new Error("Invalid servers.json structure: 'mcpServers' property is missing or invalid.");
    }
    console.log(`Loaded ${Object.keys(serversConfig.mcpServers).length} server definitions.`);
} catch (error) {
    console.error("Failed to load servers.json:", error);
    process.exit(1);
}

// multi line string
const systemPrompt = `Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within \<thinking>\</thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.`;

// --- Connection Management ---
interface ActiveConnection {
    clientId: string;
    res: Response;
    client: MCPClientStreaming;
    heartbeatInterval: NodeJS.Timeout | null;
}
const connections = new Map<string, ActiveConnection>();

// --- Middleware ---
app.use(cors());
app.use(express.json());

// --- SSE Endpoint ---
app.get('/chat', (req: Request, res: Response) => { // Line ~68 in previous full example
    const clientId = uuidv4(); // 1. ID Generated
    console.log(`[Server] Client connected, assigning ID: ${clientId}`); // 2. Logged connection

    // Set SSE headers (Potential issue point 1: Error setting headers?)
    try { // Add try-catch around header setting
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders(); // Send headers immediately
        console.log(`[${clientId}] Headers flushed successfully.`);
    } catch (headerError) {
        console.error(`[${clientId}] !!! ERROR setting/flushing headers:`, headerError);
        // If headers fail, we can't proceed with SSE. Clean up.
        res.end(); // Try to close the response
        // Don't try to add to connections map or create client if headers fail
        return;
    }

    // Function to send events (Potential issue point 2: Error defining function?)
    // This definition itself shouldn't error, but let's keep it in mind.
    const sendEvent: SendEventCallback = (event: string, data: any) => {
        console.log(`[${clientId}] Attempting to send event: ${event}. Connection exists: ${connections.has(clientId)}`);
        if (connections.has(clientId)) {
            const ssePayload: SseEvent = { type: event as SseEvent['type'], data };
            try {
                console.log(`[${clientId}] ---> Writing SSE Payload for event: ${event}`);
                res.write(formatSseEvent(ssePayload));
                console.log(`[${clientId}] <--- Successfully wrote SSE Payload for event: ${event}`);
            } catch (error) {
                console.error(`[${clientId}] Error writing to SSE stream for event ${event}:`, error);
                cleanupConnection(clientId);
                if (!res.writableEnded) res.end();
            }
        } else {
            console.warn(`[${clientId}] Cannot send event '${event}': Connection no longer exists in map.`);
        }
    };

    // Create MCPClient (Potential issue point 3: Error during constructor?)
    let mcpClient: MCPClientStreaming | null = null;
    try {
        mcpClient = new MCPClientStreaming(clientId, sendEvent, systemPrompt);
        console.log(`[${clientId}] MCPClientStreaming instance created successfully in route handler.`);
    } catch (clientError) {
        console.error(`[${clientId}] !!! ERROR creating MCPClientStreaming instance:`, clientError);
        // If client creation fails, send error via SSE if possible, then end.
        try { res.write(formatSseEvent({ type: 'error', data: { message: 'Server setup error during client init.' } })); } catch { }
        res.end();
        return;
    }

    // Store connection (Potential issue point 4: Error with Map?)
    let connectionData: ActiveConnection | null = null;
    try {
        connectionData = { clientId, res, client: mcpClient, heartbeatInterval: null }; // mcpClient is guaranteed non-null here
        connections.set(clientId, connectionData);
        console.log(`[${clientId}] Connection added to map. Map size: ${connections.size}`);
    } catch (mapError) {
        console.error(`[${clientId}] !!! ERROR adding connection to map:`, mapError);
        try { res.write(formatSseEvent({ type: 'error', data: { message: 'Server setup error during connection storage.' } })); } catch { }
        res.end();
        return;
    }


    // Send the client its ID (Potential issue point 5: Error just before/during call?)
    console.log(`[${clientId}] About to call sendEvent for 'clientId'.`);
    try { // Add try-catch around the specific call
        sendEvent('clientId', { clientId });
        console.log(`[${clientId}] Finished calling sendEvent for 'clientId'.`);
    } catch (sendIdError) {
        console.error(`[${clientId}] !!! ERROR calling sendEvent for 'clientId':`, sendIdError);
        // Cleanup because we can't even send the ID
        cleanupConnection(clientId); // cleanupConnection removes from map
        if (!res.writableEnded) res.end();
        return; // Stop processing this request
    }


    // Start MCP server connections asynchronously (Potential issue point 6: Sync error in init call?)
    // Ensure connectionData is not null before accessing .client
    if (connectionData) {
        console.log(`[${clientId}] About to call mcpClient.initializeServers.`);
        mcpClient.initializeServers(serversConfig) // Use the non-null mcpClient
            .then(() => {
                console.log(`[${clientId}] mcpClient.initializeServers promise resolved.`);
            })
            .catch(err => {
                console.error(`[${clientId}] Error during async initial server setup:`, err);
                // Send error via SSE, but connection might already be closed
                sendEvent('error', { message: "Failed during initial tool server connection.", details: err instanceof Error ? err.message : String(err) });
            });
        console.log(`[${clientId}] Finished calling mcpClient.initializeServers (async).`);

        // Heartbeat (Potential issue point 7: Error setting interval?)
        try {
            connectionData.heartbeatInterval = setInterval(() => {
                // Use a try-catch inside interval in case sendEvent fails later
                try {
                    sendEvent('pong', { timestamp: Date.now() });
                } catch (pongError) {
                    console.error(`[${clientId}] Error sending pong:`, pongError);
                    // Consider cleaning up connection if pong fails repeatedly
                }
            }, 15000);
            console.log(`[${clientId}] Heartbeat interval set.`);
        } catch (intervalError) {
            console.error(`[${clientId}] !!! ERROR setting heartbeat interval:`, intervalError);
            // Don't necessarily kill the connection for this, but log it.
        }

    } else {
        // This case should not happen if map storage succeeded, but log defensively
        console.error(`[${clientId}] !!! CRITICAL: connectionData is null after storing in map. This should not happen.`);
        if (!res.writableEnded) res.end();
        cleanupConnection(clientId); // Attempt cleanup
        return;
    }


    // Handle client disconnect
    req.on('close', () => {
        console.log(`[${clientId}] req.on('close') event triggered.`);
        cleanupConnection(clientId);
    });

    // Handle errors on the response stream itself
    res.on('error', (err) => { // Error type needed? Check imports
        console.error(`[${clientId}] res.on('error') event triggered:`, err);
        cleanupConnection(clientId);
    });

    console.log(`[${clientId}] Finished setting up route handler.`);
});


// --- Message Sending Endpoint ---
// **** Reverted to simplest inline async handler signature ****
app.post('/chat/:clientId', (async (req, res) => {
    const { clientId } = req.params as ParamsDictionary;
    const { query } = req.body as { query?: string };

    // --- Basic Validation ---
    if (!clientId || typeof clientId !== 'string') {
        return res.status(400).json({ error: "Missing or invalid clientId parameter" });
    }
    if (!query || typeof query !== 'string') {
        return res.status(400).json({ error: "Missing or invalid 'query' in request body" });
    }

    // --- Check Connection ---
    const connection = connections.get(clientId);
    if (!connection) {
        return res.status(404).json({ error: `Client ID ${clientId} not found or disconnected.` });
    }

    // --- Check Lock ---
    if (clientLocks.get(clientId)) {
        console.warn(`[${clientId}] Received query while already processing. Rejecting.`);
        return res.status(429).json({ message: "Processing previous request" });
    }

    console.log(`[${clientId}] Received query:`, query.substring(0, 100) + '...');

    try {
        // --- Acquire Lock ---
        clientLocks.set(clientId, true);
        console.log(`[${clientId}] Processing lock acquired.`);

        // --- Initiate Async Processing ---
        connection.client.processQuery(query)
            .catch(err => {
                // Handles ASYNC errors from processQuery
                console.error(`[${clientId}] Error during async processQuery execution:`, err);
                // Attempt to send error via SSE (best effort)
                try {
                    const sendEventCallback = (connection.client as any).sendEvent as SendEventCallback | undefined;
                    if (sendEventCallback && connections.has(clientId)) { // Check connection still exists
                        sendEventCallback('error', { message: "Server error during query processing.", details: err instanceof Error ? err.message : String(err) });
                    }
                } catch (sseError) {
                    console.error(`[${clientId}] Failed to send async error via SSE:`, sseError);
                }
            })
            .finally(() => {
                // --- Release Lock (Async Path) ---
                // This runs *after* the async processQuery promise settles (resolves or rejects)
                clientLocks.set(clientId, false);
                console.log(`[${clientId}] Processing lock released (async finally).`);
            });

        // --- Respond 202 Accepted ---
        // We respond immediately after successfully *starting* the async operation
        res.status(202).json({ message: "Query accepted, processing started." });

    } catch (syncError) {
        // --- Handle SYNCHRONOUS errors (e.g., failed to start processQuery) ---
        console.error(`[${clientId}] !!! SYNC ERROR initiating processQuery:`, syncError);

        // --- Release Lock (Sync Error Path) ---
        clientLocks.set(clientId, false); // *** CRITICAL FIX ***
        console.log(`[${clientId}] Processing lock released (sync error).`);

        // Respond with 500 Internal Server Error
        // Check headersSent just in case, though unlikely for sync errors here
        if (!res.headersSent) {
            return res.status(500).json({ error: "Failed to initiate query processing due to server error." });
        }
    }
}) as RequestHandler);

// --- Cleanup Function ---
function cleanupConnection(clientId: string) {
    const connection = connections.get(clientId);
    if (connection) {
        if (connection.heartbeatInterval) {
            clearInterval(connection.heartbeatInterval);
            connection.heartbeatInterval = null;
        }
        connections.delete(clientId);
        connection.client.cleanup()
            .catch(err => console.error(`[${clientId}] Error during MCP client cleanup:`, err));
        if (!connection.res.writableEnded) {
            console.log(`[${clientId}] Ending residual response stream during cleanup.`);
            connection.res.end();
        }
    } else {
        console.warn(`[Server] cleanupConnection called for non-existent clientId: ${clientId}`);
    }
}

// --- Global Error Handler ---
app.use((err: Error, req: Request, res: Response, next: NextFunction) => { // Keep types here
    console.error("[Server] Unhandled Error in middleware:", err);
    if (!res.headersSent) {
        res.status(500).json({ error: "Internal Server Error" });
    } else {
        console.error("[Server] Error occurred after headers were sent. Cannot send JSON response.");
        if (!res.writableEnded) {
            res.end();
        }
    }
});

// --- Server Shutdown Handling ---
let server: ReturnType<typeof app.listen>;

async function gracefulShutdown(signal: string) {
    console.log(`\n[Server] Received ${signal}. Shutting down gracefully...`);

    if (server) {
        server.close(async (err) => {
            if (err) {
                console.error("[Server] Error closing HTTP server:", err);
            } else {
                console.log("[Server] HTTP server closed.");
            }

            console.log(`[Server] Cleaning up ${connections.size} active connections...`);
            const cleanupPromises = Array.from(connections.keys()).map(clientId => {
                console.log(`[Server] Requesting cleanup for client: ${clientId}`);
                const connection = connections.get(clientId);
                if (connection) {
                    cleanupConnection(clientId);
                }
                return Promise.resolve();
            });

            try {
                await Promise.allSettled(cleanupPromises);
                console.log("[Server] All connection cleanup attempts finished.");
            } catch (cleanupError) {
                console.error("[Server] Error during bulk cleanup:", cleanupError);
            }

            console.log("[Server] Shutdown complete. Exiting.");
            process.exit(err ? 1 : 0);
        });

        setTimeout(() => {
            console.error('[Server] Could not close connections in time, forcing shutdown.');
            process.exit(1);
        }, 15000);
    } else {
        console.log("[Server] Server not initialized. Exiting.");
        process.exit(0);
    }
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// --- Start Server ---
server = app.listen(PORT, () => {
    console.log(`[Server] SSE Proxy Server listening on http://localhost:${PORT}`);
    console.log(`[Server] SSE endpoint: GET http://localhost:${PORT}/chat`);
    console.log(`[Server] Message endpoint: POST http://localhost:${PORT}/chat/{clientId}`);
});