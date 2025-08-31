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
}

export const ConversationalChat: React.FC = () => {
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

  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new WebSocketService();
    wsRef.current = ws;

    // Set up message handlers
    ws.on('connected', (data) => {
      setIsConnected(true);
      setConversationId(data.connection_id);
      addSystemMessage('Connected to assistant. How can I help you today?');
    });

    ws.on('conversational_update', handleConversationalUpdate);
    ws.on('typing', () => setIsTyping(true));
    ws.on('error', handleError);

    // Connect
    ws.connect().catch(console.error);

    return () => {
      ws.disconnect();
    };
  }, []);

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
        
      case 'chat_chunk':
        appendToLastMessage(update.content);
        break;
        
      case 'chat_complete':
        setIsTyping(false);
        break;
        
      case 'error':
        handleError({ error: update.content });
        break;
        
      default:
        console.log('Unhandled update type:', update.type);
    }
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

      <CardContent className="flex-1 flex flex-col p-0">
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
                  } rounded-lg p-3`}
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
                      
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      
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
                      
                      {message.briefId && (
                        <Badge variant="outline" className="mt-2">
                          Research Brief: {message.briefId.slice(0, 8)}
                        </Badge>
                      )}
                    </div>
                  </div>
                  
                  {message.status === 'sending' && (
                    <Loader2 className="w-3 h-3 animate-spin mt-1" />
                  )}
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-secondary rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <Bot className="w-4 h-4" />
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-current rounded-full animate-bounce" />
                      <span className="w-2 h-2 bg-current rounded-full animate-bounce delay-100" />
                      <span className="w-2 h-2 bg-current rounded-full animate-bounce delay-200" />
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {currentAction && (
              <Alert>
                <Loader2 className="h-4 w-4 animate-spin" />
                <AlertDescription>
                  <div className="space-y-2">
                    <p>{currentAction}</p>
                    <Progress value={actionProgress} className="h-2" />
                  </div>
                </AlertDescription>
              </Alert>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <div className="border-t p-4">
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="shrink-0"
              title="Attach file"
            >
              <Paperclip className="w-4 h-4" />
            </Button>
            
            <Input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder={
                waitingFor === 'clarification'
                  ? 'Please provide clarification...'
                  : waitingFor === 'confirmation'
                  ? 'Confirm or modify the action...'
                  : 'Ask me anything - research, CRM updates, project tasks...'
              }
              className="flex-1"
              disabled={!isConnected}
            />
            
            <Button
              variant="ghost"
              size="icon"
              className="shrink-0"
              title="Voice input"
            >
              <Mic className="w-4 h-4" />
            </Button>
            
            <Button
              onClick={sendMessage}
              disabled={!input.trim() || !isConnected}
              className="shrink-0"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
          
          <div className="mt-2 flex gap-2 flex-wrap">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInput('Research market trends for SaaS')}
            >
              <Search className="w-3 h-3 mr-1" />
              Research
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInput('Create a task for follow-up')}
            >
              <FileText className="w-3 h-3 mr-1" />
              Create Task
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInput('Schedule a meeting for tomorrow')}
            >
              <Calendar className="w-3 h-3 mr-1" />
              Schedule
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInput('Generate a report of our conversation')}
            >
              <TrendingUp className="w-3 h-3 mr-1" />
              Report
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
