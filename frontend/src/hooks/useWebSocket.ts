import { useState, useEffect, useRef } from 'react';

interface UseWebSocketOptions {
  /** Maximum reconnection attempts before giving up. */
  maxRetries?: number;
  /** Initial delay in ms before first reconnect attempt. */
  initialDelay?: number;
  /** Maximum delay in ms between reconnect attempts. */
  maxDelay?: number;
}

type ConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected' | 'failed';

/**
 * WebSocket with exponential-backoff reconnects.
 *
 * Every message is delivered to `onMessage` (via a ref, so the callback can
 * change without reconnecting). Earlier versions exposed only the *last*
 * message as React state, which silently dropped lines whenever several
 * arrived before the next render (e.g. the history replay on connect).
 */
export function useWebSocket(
  url: string | null | undefined,
  onMessage: (data: string) => void,
  options: UseWebSocketOptions = {}
) {
  const { maxRetries = 10, initialDelay = 1000, maxDelay = 30000 } = options;
  const [status, setStatus] = useState<ConnectionState>('disconnected');
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    if (!url) {
      setStatus('disconnected');
      return;
    }
    let ws: WebSocket | null = null;
    let retries = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let disposed = false;

    const scheduleReconnect = () => {
      if (disposed) return;
      if (retries >= maxRetries) {
        setStatus('failed');
        return;
      }
      const delay = Math.min(initialDelay * 2 ** retries, maxDelay);
      retries += 1;
      setStatus('reconnecting');
      timer = setTimeout(connect, delay);
    };

    function connect() {
      if (disposed) return;
      setStatus(retries === 0 ? 'connecting' : 'reconnecting');
      try {
        ws = new WebSocket(url as string);
      } catch {
        scheduleReconnect();
        return;
      }
      ws.onopen = () => {
        retries = 0;
        setStatus('connected');
      };
      ws.onmessage = (event) => onMessageRef.current(event.data);
      ws.onclose = (event) => {
        ws = null;
        if (disposed) return;
        if (event.code === 4401) {
          setStatus('failed'); // invalid API key - retrying won't help
          return;
        }
        scheduleReconnect();
      };
    }

    connect();
    return () => {
      disposed = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, [url, maxRetries, initialDelay, maxDelay]);

  return { status };
}
