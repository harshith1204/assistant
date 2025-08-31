import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { User, Bot, Search, BarChart3, Brain, Clock } from "lucide-react";
import { Message } from "../ChatInterface";
import { PhaseDetailsModal } from "./PhaseDetailsModal";

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const [selectedPhase, setSelectedPhase] = useState<"research" | "analysis" | "thinking" | null>(null);
  const formatDuration = (ms: number) => {
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case "research":
        return <Search className="w-3 h-3" />;
      case "analysis":
        return <BarChart3 className="w-3 h-3" />;
      case "thinking":
        return <Brain className="w-3 h-3" />;
      default:
        return null;
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case "research":
        return "research";
      case "analysis":
        return "analysis";
      case "thinking":
        return "thinking";
      default:
        return "muted";
    }
  };

  return (
    <div className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
      {message.role === "assistant" && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-primary flex items-center justify-center">
          <Bot className="w-4 h-4 text-research" />
        </div>
      )}
      
      <div className={`max-w-[80%] ${message.role === "user" ? "order-first" : ""}`}>
        <Card className={`p-4 ${
          message.role === "user" 
            ? "bg-research/10 border-research/20 ml-auto" 
            : "bg-card/50 border-border/50"
        }`}>
          {/* Phase indicators for assistant messages */}
          {message.role === "assistant" && message.phases && (
            <div className="flex flex-wrap gap-2 mb-3 pb-3 border-b border-border/30">
              {Object.entries(message.phases).map(([phase, status]) => {
                if (status.status === "idle") return null;
                
                return (
                  <Badge
                    key={phase}
                    variant="outline"
                    onClick={() => status.status === "complete" ? setSelectedPhase(phase as "research" | "analysis" | "thinking") : undefined}
                    className={`text-xs capitalize border-${getPhaseColor(phase)}/30 bg-${getPhaseColor(phase)}-bg text-${getPhaseColor(phase)} ${
                      status.status === "complete" ? "cursor-pointer hover:bg-opacity-80 transition-colors" : ""
                    }`}
                  >
                    <div className="flex items-center gap-1">
                      {getPhaseIcon(phase)}
                      {phase}
                      {status.status === "active" && (
                        <div className="flex gap-0.5 ml-1">
                          <div className="w-1 h-1 bg-current rounded-full animate-pulse"></div>
                          <div className="w-1 h-1 bg-current rounded-full animate-pulse delay-100"></div>
                          <div className="w-1 h-1 bg-current rounded-full animate-pulse delay-200"></div>
                        </div>
                      )}
                      {status.status === "complete" && status.duration && (
                        <div className="flex items-center gap-1 ml-1 text-muted-foreground">
                          <Clock className="w-2 h-2" />
                          <span className="text-[10px]">{formatDuration(status.duration)}</span>
                        </div>
                      )}
                    </div>
                  </Badge>
                );
              })}
            </div>
          )}
          
          <p className="text-foreground leading-relaxed">{message.content}</p>
          
          <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/30">
            <span className="text-xs text-muted-foreground">
              {message.timestamp.toLocaleTimeString()}
            </span>
          </div>
        </Card>
      </div>
      
      {message.role === "user" && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
          <User className="w-4 h-4 text-foreground" />
        </div>
      )}
      
      {/* Phase Details Modal */}
      <PhaseDetailsModal
        message={message}
        selectedPhase={selectedPhase}
        onClose={() => setSelectedPhase(null)}
      />
    </div>
  );
};