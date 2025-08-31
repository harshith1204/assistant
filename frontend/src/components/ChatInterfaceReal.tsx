import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Settings, Briefcase, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ChatMessage } from "./chat/ChatMessage";
import { ChatSettings } from "./chat/ChatSettings";
import { StreamingIndicator } from "./chat/StreamingIndicator";
import { WebSocketService, apiService } from "@/services/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  phases?: {
    research?: { 
      status: "active" | "complete" | "idle"; 
      duration?: number;
      details?: {
        sources: string[];
        queries: string[];
        findings: string[];
      };
    };
    analysis?: { 
      status: "active" | "complete" | "idle"; 
      duration?: number;
      details?: {
        dataPoints: string[];
        patterns: string[];
        insights: string[];
      };
    };
    thinking?: { 
      status: "active" | "complete" | "idle"; 
      duration?: number;
      details?: {
        reasoning: string[];
        considerations: string[];
        conclusions: string[];
      };
    };
  };
  sources?: Array<{
    title: string;
    url: string;
    snippet: string;
  }>;
  metadata?: Record<string, any>;
}

export const ChatInterfaceReal = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [researchEnabled, setResearchEnabled] = useState(true);
  const [selectedModel, setSelectedModel] = useState("moonshotai/kimi-k2-instruct");
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [wsService, setWsService] = useState<WebSocketService | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "connecting" | "disconnected">("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState<string>("");
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStreamingMessage]);

  // Initialize WebSocket connection
  useEffect(() => {
    const initWebSocket = async () => {
      setConnectionStatus("connecting");
      const ws = new WebSocketService();
      
      try {
        await ws.connect();
        
        // Register message handlers
        ws.on('connected', (data) => {
          console.log('Connected:', data);
          setConnectionStatus("connected");
          setError(null);
        });

        ws.on('message', (data) => {
          // Handle complete message response
          const message: Message = {
            id: data.message_id || Date.now().toString(),
            role: "assistant",
            content: data.response,
            timestamp: new Date(),
            sources: data.sources,
            metadata: data.metadata,
          };
          
          setMessages(prev => [...prev, message]);
          setIsProcessing(false);
        });

        ws.on('stream_start', (data) => {
          console.log('Stream started:', data);
          setCurrentStreamingMessage("");
          setStreamingMessageId(Date.now().toString());
        });

        ws.on('stream_chunk', (data) => {
          setCurrentStreamingMessage(prev => prev + data.content);
        });

        ws.on('stream_end', (data) => {
          const message: Message = {
            id: streamingMessageId || Date.now().toString(),
            role: "assistant",
            content: data.full_response || currentStreamingMessage,
            timestamp: new Date(),
          };
          
          setMessages(prev => [...prev, message]);
          setCurrentStreamingMessage("");
          setStreamingMessageId(null);
          setIsProcessing(false);
        });

        ws.on('typing', () => {
          // Show typing indicator
        });

        ws.on('error', (data) => {
          setError(data.error || 'An error occurred');
          setIsProcessing(false);
        });

        ws.on('research_status', (data) => {
          console.log('Research status:', data);
          // Update research phase status
          if (streamingMessageId) {
            setMessages(prev => prev.map(msg => 
              msg.id === streamingMessageId 
                ? { 
                    ...msg, 
                    phases: { 
                      ...msg.phases, 
                      research: { 
                        status: data.status === 'started' ? 'active' : 'complete' 
                      } 
                    } 
                  }
                : msg
            ));
          }
        });

        ws.on('research_complete', (data) => {
          console.log('Research complete:', data);
          const message: Message = {
            id: data.message_id || Date.now().toString(),
            role: "assistant",
            content: data.response,
            timestamp: new Date(),
            sources: data.sources,
            metadata: data.metadata,
            phases: {
              research: { status: 'complete' }
            }
          };
          
          setMessages(prev => [...prev, message]);
          setIsProcessing(false);
        });

        setWsService(ws);
        
        // Send ping every 30 seconds to keep connection alive
        const pingInterval = setInterval(() => {
          if (ws.isConnected()) {
            ws.ping();
          }
        }, 30000);

        return () => {
          clearInterval(pingInterval);
        };
        
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setConnectionStatus("disconnected");
        setError('Failed to connect to chat service');
      }
    };

    initWebSocket();

    return () => {
      if (wsService) {
        wsService.disconnect();
      }
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isProcessing) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setIsProcessing(true);
    setError(null);

    try {
      if (wsService && wsService.isConnected()) {
        // Use WebSocket for streaming
        if (researchEnabled) {
          wsService.sendResearchRequest(inputValue, conversationId);
        } else {
          wsService.sendChatMessage(inputValue, conversationId, true);
        }
        
        if (!conversationId) {
          // Set conversation ID after first message
          setConversationId(`conv_${Date.now()}`);
        }
      } else {
        // Fallback to HTTP API
        const response = await apiService.sendChatMessage({
          message: inputValue,
          conversation_id: conversationId,
          use_web_search: researchEnabled,
          model: selectedModel,
        });

        const assistantMessage: Message = {
          id: response.message_id,
          role: "assistant",
          content: response.response,
          timestamp: new Date(),
          sources: response.sources,
          metadata: response.metadata,
        };

        setMessages(prev => [...prev, assistantMessage]);
        setConversationId(response.conversation_id);
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setError('Failed to send message. Please try again.');
      setIsProcessing(false);
    }
  };

  const loadConversationHistory = useCallback(async (convId: string) => {
    try {
      if (wsService && wsService.isConnected()) {
        wsService.getConversationHistory(convId);
      } else {
        const conversation = await apiService.getConversation(convId);
        const loadedMessages: Message[] = conversation.messages.map((msg: any) => ({
          id: msg.message_id,
          role: msg.role,
          content: msg.content,
          timestamp: new Date(msg.timestamp),
          metadata: msg.metadata,
        }));
        setMessages(loadedMessages);
        setConversationId(convId);
      }
    } catch (error) {
      console.error('Failed to load conversation:', error);
      setError('Failed to load conversation history');
    }
  }, [wsService]);

  return (
    <div className="flex h-screen bg-gradient-chat">
      <div className="flex-1 flex flex-col max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-primary">
              <Briefcase className="w-6 h-6 text-research" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-foreground">AI Business Assistant</h1>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>Model: {selectedModel.split('/').pop()}</span>
                <span>•</span>
                <span>Research: {researchEnabled ? "Active" : "Disabled"}</span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <div className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-green-500' : 
                    connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 
                    'bg-red-500'
                  }`} />
                  {connectionStatus}
                </span>
              </div>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSettings(!showSettings)}
            className="border-border/50"
          >
            <Settings className="w-4 h-4" />
          </Button>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <ChatSettings
            researchEnabled={researchEnabled}
            setResearchEnabled={setResearchEnabled}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            onClose={() => setShowSettings(false)}
          />
        )}

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive" className="m-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="mb-4 p-4 rounded-full bg-gradient-primary w-16 h-16 mx-auto flex items-center justify-center">
                <Briefcase className="w-8 h-8 text-research" />
              </div>
              <h2 className="text-xl font-semibold text-foreground mb-2">
                Welcome to AI Business Assistant
              </h2>
              <p className="text-muted-foreground">
                Get business insights, analysis, and strategic advice powered by AI research.
              </p>
            </div>
          )}
          
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          
          {/* Show streaming message */}
          {currentStreamingMessage && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-primary flex items-center justify-center">
                <Briefcase className="w-4 h-4 text-research" />
              </div>
              <div className="flex-1 space-y-2">
                <div className="text-sm text-muted-foreground">Assistant</div>
                <div className="prose prose-sm max-w-none">
                  {currentStreamingMessage}
                  <span className="inline-block w-2 h-4 bg-foreground/50 animate-pulse ml-1" />
                </div>
              </div>
            </div>
          )}
          
          {isProcessing && !currentStreamingMessage && <StreamingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <Card className="m-4 bg-card/50 backdrop-blur-sm border-border/50">
          <form onSubmit={handleSubmit} className="flex gap-2 p-4">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={connectionStatus === 'connected' ? "Ask me anything..." : "Connecting to service..."}
              disabled={isProcessing || connectionStatus !== 'connected'}
              className="flex-1 bg-background/50 border-border/50 focus:border-research focus:ring-research/20"
            />
            <Button 
              type="submit" 
              disabled={!inputValue.trim() || isProcessing || connectionStatus !== 'connected'}
              className="bg-research hover:bg-research/90 text-white shadow-glow"
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </form>
        </Card>
      </div>
    </div>
  );
};