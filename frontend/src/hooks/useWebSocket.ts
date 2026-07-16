import { useState, useEffect, useRef, useCallback } from 'react';

interface UseWebSocketOptions {
  /** Maximum reconnection attempts before giving up. */
  maxRetries?: number;
  /** Initial delay in ms before first reconnect attempt. */
  initialDelay?: number;
  /** Maximum delay in ms between reconnect attempts. */
  maxDelay?: number;
}

type ConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected' | 'failed';

export function useWebSocket(
  url: string | null | undefined,
  options: UseWebSocketOptions = {}
) {
  const { maxRetries = 10, initialDelay = 1000, maxDelay = 30000 } = options;
  const [status, setStatus] = useState<ConnectionState>('disconnected');
  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const urlRef = useRef(url);

  // Keep url ref current
  useEffect(() => { urlRef.current = url ?? null; }, [url]);

  const connect = useCallback(() => {
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (!urlRef.current) {
      setStatus('disconnected');
      return;
    }

    setStatus(prev => prev === 'connected' ? 'connected' : retriesRef.current === 0 ? 'connecting' : 'reconnecting');

    try {
      const ws = new WebSocket(urlRef.current);
      wsRef.current = ws;

      ws.onopen = () => {
        retriesRef.current = 0;
        setStatus('connected');
      };

      ws.onmessage = (event) => {
        setLastMessage(event.data);
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (retriesRef.current < maxRetries && urlRef.current) {
          const delay = Math.min(initialDelay * Math.pow(2, retriesRef.current), maxDelay);
          retriesRef.current++;
          setStatus('reconnecting');
          timeoutRef.current = setTimeout(connect, delay);
        } else {
          setStatus(retriesRef.current >= maxRetries ? 'failed' : 'disconnected');
        }
      };

      ws.onerror = () => {
        // onclose will handle reconnection
      };
    } catch {
      // Connection failed, schedule retry
      if (retriesRef.current < maxRetries) {
        const delay = Math.min(initialDelay * Math.pow(2, retriesRef.current), maxDelay);
        retriesRef.current++;
        timeoutRef.current = setTimeout(connect, delay);
      } else {
        setStatus('failed');
      }
    }
  }, [maxRetries, initialDelay, maxDelay]);

  // Connect when url changes
  useEffect(() => {
    retriesRef.current = 0;
    if (url) {
      connect();
    } else {
      setStatus('disconnected');
    }
    return () => {
      clearTimeout(timeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [url, connect]);

  const send = useCallback((data: string | ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  const reconnect = useCallback(() => {
    retriesRef.current = 0;
    connect();
  }, [connect]);

  return { status, lastMessage, send, reconnect };
}
