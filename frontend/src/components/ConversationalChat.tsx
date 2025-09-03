import React, { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketService } from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Send, 
  Mic, 
  Paperclip, 
  Search, 
  Calendar, 
  FileText, 
  Database,
  CheckCircle,
  XCircle,
  AlertCircle,
  Loader2,
  Bot,
  User,
  Sparkles,
  TrendingUp,
  Users,
  Building
} from 'lucide-react';

interface ConversationalUpdate {
  type: string;
  content: string;
  intent?: string;
  confidence?: number;
  preview?: any;
  actions?: Array<{ label: string; value: string }>;
  questions?: string[];
  topics?: string[];
  waiting_for_response?: boolean;
  metadata?: any;
  step?: number;
  total_steps?: number;
  current_action?: string;
  brief_id?: string;
  findings_count?: number;
  ideas_count?: number;
  result?: any;
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  updates?: ConversationalUpdate[];
  status?: 'sending' | 'sent' | 'error';
  intent?: string;
  confidence?: number;
  questions?: string[];
  preview?: any;
  actions?: Array<{ label: string; value: string }>;
  briefId?: string;
}

interface ConversationalChatProps {
  userId?: string;
}

export const ConversationalChat: React.FC<ConversationalChatProps> = ({ userId }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [currentAction, setCurrentAction] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [waitingFor, setWaitingFor] = useState<string | null>(null);
  const [actionProgress, setActionProgress] = useState<number>(0);
  
  const wsRef = useRef<WebSocketService | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Business and User context
  const [businessId, setBusinessId] = useState<string>(() => localStorage.getItem('businessId') || '');
  const [effectiveUserId, setEffectiveUserId] = useState<string | undefined>(userId || localStorage.getItem('userId') || undefined);

  // Initialize WebSocket connection
  useEffect(() => {
    console.log('🔌 Initializing WebSocket connection...');
    console.log('User ID:', effectiveUserId);
    console.log('Business ID:', businessId);

    const ws = new WebSocketService(effectiveUserId, businessId);
    wsRef.current = ws;

    // Set up message handlers
    ws.on('connected', (data) => {
      setIsConnected(true);
      setConversationId(data.connection_id);
      addSystemMessage('Connected to assistant. How can I help you today?');
    });

    ws.on('conversational_update', handleConversationalUpdate);
    ws.on('typing', () => setIsTyping(true));
    ws.on('token', handleToken);  // Handle streaming tokens
    ws.on('message_complete', handleMessageComplete);  // Handle final message
    ws.on('error', handleError);

    // Connect
    console.log('🔌 Calling ws.connect()...');
    ws.connect()
      .then(() => console.log('🔌 WebSocket connect() promise resolved'))
      .catch((error) => {
        console.error('❌ WebSocket connect() promise rejected:', error);
        addSystemMessage(`Connection failed: ${error.message}`);
      });

    return () => {
      ws.disconnect();
    };
  }, [effectiveUserId, businessId]);

  // Keep localStorage in sync when IDs change
  useEffect(() => {
    if (effectiveUserId) localStorage.setItem('userId', effectiveUserId);
    if (businessId !== undefined) localStorage.setItem('businessId', businessId);
  }, [effectiveUserId, businessId]);

  const handleConversationalUpdate = (update: ConversationalUpdate) => {
    console.log('Conversational update:', update);
    
    // Handle different update types
    switch (update.type) {
      case 'acknowledgment':
        addAssistantMessage(update.content, {
          intent: update.intent,
          confidence: update.confidence
        });
        break;
        
      case 'clarification_needed':
        setWaitingFor('clarification');
        addAssistantMessage(update.content, {
          questions: update.questions,
          needsResponse: true
        });
        break;
        
      case 'confirmation_needed':
        setWaitingFor('confirmation');
        addAssistantMessage('', {
          preview: update.preview,
          actions: update.actions,
          needsConfirmation: true
        });
        break;
        
      case 'progress':
        setActionProgress((update.step || 0) / (update.total_steps || 1) * 100);
        setCurrentAction(update.current_action || null);
        break;
        
      case 'research_started':
        addSystemMessage(`🔍 Research started on: ${update.topics?.join(', ') || 'your query'}`);
        break;
        
      case 'research_complete':
        addAssistantMessage(update.content, {
          briefId: update.brief_id,
          findingsCount: update.findings_count,
          ideasCount: update.ideas_count,
          metadata: update.metadata
        });
        setActionProgress(0);
        setCurrentAction(null);
        break;
        
      case 'crm_action_complete':
      case 'pms_action_complete':
        addAssistantMessage(update.content, {
          result: update.result,
          actionType: update.type
        });
        setActionProgress(0);
        setCurrentAction(null);
        break;
        

        
      case 'error':
        handleError({ error: update.content });
        break;
        
      default:
        console.log('Unhandled update type:', update.type);
    }
  };

  const handleToken = (data: any) => {
    console.log('Received token:', data.delta);
    // If this is the first token and we're typing, create the assistant message
    if (isTyping && messages.length > 0 && messages[messages.length - 1].role !== 'assistant') {
      // Create a new assistant message for the streaming response
      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        status: 'sent'
      };
      setMessages(prev => [...prev, assistantMessage]);
    }
    appendToLastMessage(data.delta);
  };

  const handleMessageComplete = (data: any) => {
    console.log('Message complete:', data);
    setIsTyping(false);
    // The message content should already be complete from the token streaming
  };

  const handleError = (data: any) => {
    console.error('WebSocket error:', data);
    addSystemMessage(`Error: ${data.error}`, 'error');
    setIsTyping(false);
    setActionProgress(0);
    setCurrentAction(null);
  };

  const addSystemMessage = (content: string, type: string = 'info') => {
    const message: Message = {
      id: Date.now().toString(),
      role: 'system',
      content,
      timestamp: new Date(),
      status: 'sent'
    };
    setMessages(prev => [...prev, message]);
  };

  const addAssistantMessage = (content: string, metadata?: any) => {
    const message: Message = {
      id: Date.now().toString(),
      role: 'assistant',
      content,
      timestamp: new Date(),
      status: 'sent',
      ...metadata
    };
    setMessages(prev => [...prev, message]);
    setIsTyping(false);
  };

  const appendToLastMessage = (content: string) => {
    setMessages(prev => {
      const newMessages = [...prev];
      const lastMessage = newMessages[newMessages.length - 1];
      if (lastMessage && lastMessage.role === 'assistant') {
        lastMessage.content += content;
      } else {
        // Create new assistant message if needed
        newMessages.push({
          id: Date.now().toString(),
          role: 'assistant',
          content,
          timestamp: new Date(),
          status: 'sent'
        });
      }
      return newMessages;
    });
  };

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current?.isConnected()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      status: 'sending'
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    setWaitingFor(null);

    // Send via WebSocket with conversational flag
    console.log('Sending message via WebSocket:', {
      message: input,
      conversation_id: conversationId,
      conversational: true,
      stream: true
    });
    wsRef.current.send('chat', {
      message: input,
      conversation_id: conversationId,
      conversational: true,
      stream: true
    });

    // Mark as sent
    setTimeout(() => {
      setMessages(prev => 
        prev.map(msg => 
          msg.id === userMessage.id 
            ? { ...msg, status: 'sent' }
            : msg
        )
      );
    }, 100);
  };

  const handleAction = (action: string) => {
    if (action === 'confirm' && waitingFor === 'confirmation') {
      sendMessage();
    } else if (action === 'cancel') {
      setWaitingFor(null);
      addSystemMessage('Action cancelled');
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getIntentIcon = (intent?: string) => {
    switch (intent) {
      case 'research': return <Search className="w-4 h-4" />;
      case 'crm_action': return <Users className="w-4 h-4" />;
      case 'pms_action': return <FileText className="w-4 h-4" />;
      case 'meeting_scheduling': return <Calendar className="w-4 h-4" />;
      case 'report_generation': return <TrendingUp className="w-4 h-4" />;
      default: return <Sparkles className="w-4 h-4" />;
    }
  };

  const getConfidenceBadge = (confidence?: number) => {
    if (!confidence) return null;
    
    const variant = confidence > 0.8 ? 'default' : confidence > 0.5 ? 'secondary' : 'outline';
    const label = confidence > 0.8 ? 'High' : confidence > 0.5 ? 'Medium' : 'Low';
    
    return (
      <Badge variant={variant} className="ml-2">
        {label} ({Math.round(confidence * 100)}%)
      </Badge>
    );
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="border-b">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Bot className="w-5 h-5" />
            AI Assistant
          </CardTitle>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2">
              <Input
                value={businessId}
                onChange={(e) => setBusinessId(e.target.value)}
                placeholder="Business ID"
                className="w-40"
              />
              <Input
                value={effectiveUserId || ''}
                onChange={(e) => setEffectiveUserId(e.target.value || undefined)}
                placeholder="User ID"
                className="w-40"
              />
            </div>
            {isConnected ? (
              <Badge variant="default" className="bg-green-500">
                <CheckCircle className="w-3 h-3 mr-1" />
                Connected
              </Badge>
            ) : (
              <Badge variant="secondary">
                <AlertCircle className="w-3 h-3 mr-1" />
                Connecting...
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-0 min-h-0">
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[70%] ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : message.role === 'system'
                      ? 'bg-muted'
                      : 'bg-secondary'
                  } rounded-lg p-3 overflow-hidden`}
                >
                  <div className="flex items-start gap-2">
                    {message.role === 'assistant' && <Bot className="w-4 h-4 mt-1" />}
                    {message.role === 'user' && <User className="w-4 h-4 mt-1" />}
                    <div className="flex-1">
                      {message.intent && (
                        <div className="flex items-center mb-2">
                          {getIntentIcon(message.intent)}
                          <span className="ml-1 text-sm font-medium capitalize">
                            {message.intent.replace('_', ' ')}
                          </span>
                          {getConfidenceBadge(message.confidence)}
                        </div>
                      )}
                      
                      <div className="whitespace-pre-wrap break-words">{message.content}</div>
                      
                      {message.questions && (
                        <div className="mt-3 space-y-2">
                          <p className="text-sm font-medium">Please clarify:</p>
                          {message.questions.map((q: string, i: number) => (
                            <Button
                              key={i}
                              variant="outline"
                              size="sm"
                              className="w-full justify-start"
                              onClick={() => setInput(q)}
                            >
                              {q}
                            </Button>
                          ))}
                        </div>
                      )}
                      
                      {message.preview && (
                        <Card className="mt-3">
                          <CardContent className="p-3">
                            <h4 className="font-medium mb-2">Action Preview:</h4>
                            <p className="text-sm">{message.preview.description}</p>
                            <div className="mt-2 flex gap-2">
                              {message.actions?.map((action: any) => (
                                <Button
                                  key={action.value}
                                  size="sm"
                                  variant={action.value === 'confirm' ? 'default' : 'outline'}
                                  onClick={() => handleAction(action.value)}
                                >
                                  {action.label}
                                </Button>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {currentAction && (
                        <div className="mt-3">
                          <p className="text-sm font-medium">{currentAction}</p>
                          <Progress value={actionProgress} className="h-2" />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <div className="p-4 border-t">
          <div className="flex gap-2 items-center">
            <Button variant="outline" size="icon">
              <Mic className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="icon">
              <Paperclip className="w-4 h-4" />
            </Button>
            <Input
              ref={inputRef}
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            />
            <Button onClick={sendMessage} disabled={!isConnected || !input.trim()}>
              <Send className="w-4 h-4 mr-2" />
              Send
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
