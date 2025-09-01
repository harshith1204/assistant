/**
 * API Service for communication with backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export interface ChatMessage {
  message: string;
  conversation_id?: string;
  user_id?: string;
  business_id?: string;
  use_web_search?: boolean;
  stream?: boolean;
  model?: string;
  temperature?: number;
  max_tokens?: number;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  message_id: string;
  research_brief_id?: string;
  sources?: Array<{
    title: string;
    url: string;
    snippet: string;
  }>;
  metadata?: Record<string, any>;
}

export interface ResearchRequest {
  query: string;
  max_sources?: number;
  credibility_threshold?: number;
  recency_months?: number;
  use_cache?: boolean;
}

export interface ResearchBrief {
  brief_id: string;
  query: string;
  date: string;
  findings: Array<{
    finding: string;
    source: string;
    credibility_score: number;
    relevance_score: number;
    date: string;
    citation_url: string;
  }>;
  ideas: Array<{
    idea: string;
    impact: string;
    feasibility: string;
    time_estimate: string;
    rice_score: number;
    resources: string[];
    risks: string[];
  }>;
  executive_summary: string;
  total_sources: number;
  average_confidence: number;
  processing_time: number;
}

class ApiService {
  private headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  /**
   * Send a chat message
   */
  async sendChatMessage(message: ChatMessage): Promise<ChatResponse> {
    // Ensure a stable user_id for long-term personalization
    let userId = message.user_id || localStorage.getItem('userId') || undefined;
    if (!userId) {
      userId = `anon_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('userId', userId);
    }

    const payload = {
      ...message,
      user_id: userId,
    };

    const response = await fetch(`${API_BASE_URL}/chat/message`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Chat request failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Run research
   */
  async runResearch(request: ResearchRequest): Promise<ResearchBrief> {
    const response = await fetch(`${API_BASE_URL}/research/run`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Research request failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get research brief by ID
   */
  async getResearchBrief(briefId: string): Promise<ResearchBrief> {
    const response = await fetch(`${API_BASE_URL}/research/brief/${briefId}`, {
      method: 'GET',
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`Failed to get research brief: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * List conversations
   */
  async listConversations(userId?: string, limit = 20, offset = 0) {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });
    
    if (userId) {
      params.append('user_id', userId);
    }

    const response = await fetch(`${API_BASE_URL}/chat/conversations?${params}`, {
      method: 'GET',
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`Failed to list conversations: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get conversation by ID
   */
  async getConversation(conversationId: string) {
    const response = await fetch(`${API_BASE_URL}/chat/conversation/${conversationId}`, {
      method: 'GET',
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`Failed to get conversation: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Delete conversation
   */
  async deleteConversation(conversationId: string) {
    const response = await fetch(`${API_BASE_URL}/chat/conversation/${conversationId}`, {
      method: 'DELETE',
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`Failed to delete conversation: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get memory stats
   */
  async getMemoryStats(userId?: string) {
    const params = userId ? `?user_id=${userId}` : '';
    const response = await fetch(`${API_BASE_URL}/chat/memory/stats${params}`, {
      method: 'GET',
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`Failed to get memory stats: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Health check
   */
  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }
}

export const apiService = new ApiService();

/**
 * WebSocket Service for real-time communication
 */
export class WebSocketService {
  private ws: WebSocket | null = null;
  private connectionId: string;
  private userId: string;
  private businessId: string;
  private messageHandlers: Map<string, (data: any) => void> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(userId?: string, businessId?: string) {
    this.connectionId = this.generateConnectionId();
    // Ensure we always have a user_id for dual context
    this.userId = this.getOrCreateUserId(userId);
    // Ensure we always have a business_id for scoping
    this.businessId = this.getOrCreateBusinessId(businessId);
  }

  /**
   * Get or create a persistent user ID
   */
  private getOrCreateUserId(providedUserId?: string): string {
    if (providedUserId) {
      // Store provided user ID for persistence
      localStorage.setItem('userId', providedUserId);
      return providedUserId;
    }

    // Check localStorage for existing user ID
    const storedUserId = localStorage.getItem('userId');
    if (storedUserId) {
      return storedUserId;
    }

    // Generate anonymous user ID
    const anonUserId = `anon_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('userId', anonUserId);
    console.log('Generated anonymous user ID:', anonUserId);
    return anonUserId;
  }

  /**
   * Get or create a persistent business ID
   */
  private getOrCreateBusinessId(providedBusinessId?: string): string {
    if (providedBusinessId) {
      localStorage.setItem('businessId', providedBusinessId);
      return providedBusinessId;
    }
    const storedBusinessId = localStorage.getItem('businessId');
    if (storedBusinessId) {
      return storedBusinessId;
    }
    // Business ID can be optional; keep empty string if not set
    return '';
  }

  private generateConnectionId(): string {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Connect to WebSocket with dual context support
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Always include user_id for dual context
      const params = new URLSearchParams();
      if (this.userId) params.set('user_id', this.userId);
      if (this.businessId) params.set('business_id', this.businessId);
      const wsUrl = `${WS_BASE_URL}/ws/chat/${this.connectionId}?${params.toString()}`;
      
      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.handleDisconnect();
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Handle incoming message
   */
  private handleMessage(message: any) {
    const { type, data } = message;
    
    // Handle session info to update user_id if needed
    if (type === 'session_info') {
      const { user_id, business_id } = data;
      if (user_id && user_id !== this.userId) {
        this.userId = user_id;
        localStorage.setItem('userId', user_id);
        console.log('Updated user ID from server:', user_id);
      }
      if (typeof business_id === 'string' && business_id !== this.businessId) {
        this.businessId = business_id;
        localStorage.setItem('businessId', business_id);
        console.log('Updated business ID from server:', business_id);
      }
    }
    
    const handler = this.messageHandlers.get(type);
    
    if (handler) {
      handler(data);
    } else {
      console.warn(`No handler for message type: ${type}`);
    }
  }

  /**
   * Handle disconnection and reconnect
   */
  private handleDisconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.connect().catch(console.error);
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  /**
   * Register message handler
   */
  on(type: string, handler: (data: any) => void) {
    this.messageHandlers.set(type, handler);
  }

  /**
   * Remove message handler
   */
  off(type: string) {
    this.messageHandlers.delete(type);
  }

  /**
   * Send message
   */
  send(type: string, data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  /**
   * Send chat message with dual context
   */
  sendChatMessage(message: string, conversationId?: string, stream = true) {
    // Ensure conversation_id is provided for dual context
    const finalConversationId = conversationId || this.generateNewConversationId();
    
    this.send('chat', {
      message,
      conversation_id: finalConversationId,
      stream,
      user_id: this.userId,
      business_id: this.businessId,
    });
    
    return finalConversationId;  // Return for UI to track
  }
  
  /**
   * Generate a new conversation ID
   */
  private generateNewConversationId(): string {
    return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Send research request
   */
  sendResearchRequest(query: string, conversationId?: string) {
    this.send('research', {
      query,
      conversation_id: conversationId,
    });
  }

  /**
   * Get conversation list
   */
  getConversationList() {
    this.send('list_conversations', {});
  }

  /**
   * Get conversation history
   */
  getConversationHistory(conversationId: string) {
    this.send('get_history', {
      conversation_id: conversationId,
    });
  }

  /**
   * Send ping
   */
  ping() {
    this.send('ping', {});
  }

  /**
   * Disconnect
   */
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}
