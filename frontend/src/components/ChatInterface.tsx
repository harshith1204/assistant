import { useState, useRef, useEffect } from "react";
import { Send, Settings, Briefcase } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ChatMessage } from "./chat/ChatMessage";
import { ChatSettings } from "./chat/ChatSettings";
import { StreamingIndicator } from "./chat/StreamingIndicator";

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
}

export const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [researchEnabled, setResearchEnabled] = useState(true);
  const [selectedModel, setSelectedModel] = useState("gpt-4");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const simulateAIResponse = async (userMessage: string) => {
    const messageId = Date.now().toString();
    
    // Add user message
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: userMessage,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMsg]);
    setInputValue("");
    setIsProcessing(true);

    // Create AI response with phases
    const aiMessage: Message = {
      id: messageId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      phases: {
        research: { status: "idle" },
        analysis: { status: "idle" },
        thinking: { status: "idle" },
      },
    };

    setMessages(prev => [...prev, aiMessage]);

    // Simulate research phase
    if (researchEnabled) {
      const researchStart = Date.now();
      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? { ...msg, phases: { ...msg.phases, research: { status: "active" } } }
          : msg
      ));

      await new Promise(resolve => setTimeout(resolve, 2000));

      // Generate research details
      const researchDetails = {
        sources: [
          "Industry reports from McKinsey & Company",
          "Harvard Business Review articles",
          "Financial data from Bloomberg Terminal",
          "Market research from Statista"
        ],
        queries: [
          `"${userMessage}" business trends 2024`,
          `Market analysis "${userMessage}"`,
          `Best practices "${userMessage}" enterprise`,
          `ROI metrics "${userMessage}"`
        ],
        findings: [
          "Current market trends show significant growth potential",
          "Industry leaders are adopting similar strategies",
          "Cost-benefit analysis indicates positive ROI",
          "Regulatory environment remains favorable"
        ]
      };

      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? { 
              ...msg, 
              phases: { 
                ...msg.phases, 
                research: { 
                  status: "complete", 
                  duration: Date.now() - researchStart,
                  details: researchDetails
                } 
              } 
            }
          : msg
      ));
    }

    // Simulate analysis phase
    const analysisStart = Date.now();
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, phases: { ...msg.phases, analysis: { status: "active" } } }
        : msg
    ));

    await new Promise(resolve => setTimeout(resolve, 1500));

    // Generate analysis details
    const analysisDetails = {
      dataPoints: [
        "Market size: $2.4B with 15% YoY growth",
        "Customer acquisition cost decreased by 23%",
        "Revenue per customer increased by 18%",
        "Competitive landscape analysis complete"
      ],
      patterns: [
        "Peak demand occurs during Q4 seasonality",
        "Mobile users show 40% higher engagement",
        "Enterprise clients prefer annual contracts",
        "Cross-selling opportunities identified"
      ],
      insights: [
        "Premium tier adoption rate exceeds projections",
        "Customer churn correlates with onboarding quality",
        "Geographic expansion shows strong potential",
        "Technology stack optimization needed"
      ]
    };

    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { 
            ...msg, 
            phases: { 
              ...msg.phases, 
              analysis: { 
                status: "complete", 
                duration: Date.now() - analysisStart,
                details: analysisDetails
              } 
            } 
          }
        : msg
    ));

    // Simulate thinking phase
    const thinkingStart = Date.now();
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, phases: { ...msg.phases, thinking: { status: "active" } } }
        : msg
    ));

    await new Promise(resolve => setTimeout(resolve, 1000));

    // Generate thinking details
    const thinkingDetails = {
      reasoning: [
        "Evaluated multiple strategic frameworks (Porter's Five Forces, SWOT)",
        "Considered stakeholder impact and change management requirements",
        "Assessed resource allocation and timeline constraints",
        "Analyzed risk factors and mitigation strategies"
      ],
      considerations: [
        "Budget implications for Q4 implementation",
        "Team capacity and skill requirements",
        "Competitive response and market timing",
        "Regulatory compliance and legal requirements"
      ],
      conclusions: [
        "Recommend phased implementation approach",
        "Priority focus on high-impact, low-risk initiatives",
        "Establish KPIs and monitoring framework",
        "Plan for quarterly review and optimization"
      ]
    };

    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { 
            ...msg, 
            phases: { 
              ...msg.phases, 
              thinking: { 
                status: "complete", 
                duration: Date.now() - thinkingStart,
                details: thinkingDetails
              } 
            },
            content: `I understand you're asking about "${userMessage}". Based on my ${researchEnabled ? 'research and ' : ''}analysis, here's a comprehensive response that addresses your query with relevant insights and actionable information.`
          }
        : msg
    ));

    setIsProcessing(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isProcessing) return;

    await simulateAIResponse(inputValue);
  };

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
              <p className="text-sm text-muted-foreground">
                Model: {selectedModel} • Research: {researchEnabled ? "Active" : "Disabled"}
              </p>
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
          
          {isProcessing && <StreamingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <Card className="m-4 bg-card/50 backdrop-blur-sm border-border/50">
          <form onSubmit={handleSubmit} className="flex gap-2 p-4">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask me anything..."
              disabled={isProcessing}
              className="flex-1 bg-background/50 border-border/50 focus:border-research focus:ring-research/20"
            />
            <Button 
              type="submit" 
              disabled={!inputValue.trim() || isProcessing}
              className="bg-research hover:bg-research/90 text-white shadow-glow"
            >
              <Send className="w-4 h-4" />
            </Button>
          </form>
        </Card>
      </div>
    </div>
  );
};