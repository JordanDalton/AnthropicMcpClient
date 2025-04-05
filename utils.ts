// utils.ts
import { SseEvent } from './types.js'; // Assuming types.ts is in the same directory

export function filterStringEnv(env: NodeJS.ProcessEnv | Record<string, string | undefined>): Record<string, string> {
    const result: Record<string, string> = {};
    for (const key in env) {
        const value = env[key];
        if (typeof value === 'string') {
            result[key] = value;
        }
    }
    return result;
}

export function formatSseEvent(event: SseEvent): string {
    // SSE format requires "event:" and "data:" lines, ending with "\n\n"
    const dataString = JSON.stringify(event.data);
    return `event: ${event.type}\ndata: ${dataString}\n\n`;
}

export function truncateString(str: string, maxLength: number): string {
    if (str.length <= maxLength) {
        return str;
    }
    return str.substring(0, maxLength - 15) + `...[truncated ${str.length - maxLength + 15} chars]`;
}