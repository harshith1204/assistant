import { Card } from "@/components/ui/card";
import { Bot, Search, BarChart3, Brain } from "lucide-react";

export const StreamingIndicator = () => {
  return (
    <div className="flex gap-3 justify-start">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-primary flex items-center justify-center">
        <Bot className="w-4 h-4 text-research animate-pulse" />
      </div>
      
      <Card className="max-w-[80%] p-4 bg-card/50 border-border/50">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Search className="w-4 h-4 text-research animate-pulse" />
            <BarChart3 className="w-4 h-4 text-analysis animate-pulse delay-300" />
            <Brain className="w-4 h-4 text-thinking animate-pulse delay-700" />
          </div>
          <div className="flex gap-1">
            <div className="w-2 h-2 bg-foreground/40 rounded-full animate-pulse"></div>
            <div className="w-2 h-2 bg-foreground/40 rounded-full animate-pulse delay-150"></div>
            <div className="w-2 h-2 bg-foreground/40 rounded-full animate-pulse delay-300"></div>
          </div>
        </div>
        <p className="text-sm text-muted-foreground mt-2">
          AI is processing your request...
        </p>
      </Card>
    </div>
  );
};