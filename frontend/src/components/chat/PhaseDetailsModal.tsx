import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Search, BarChart3, Brain, Clock, CheckCircle } from "lucide-react";
import { Message } from "../ChatInterface";

interface PhaseDetailsModalProps {
  message: Message;
  selectedPhase: "research" | "analysis" | "thinking" | null;
  onClose: () => void;
}

export const PhaseDetailsModal = ({ message, selectedPhase, onClose }: PhaseDetailsModalProps) => {
  if (!selectedPhase || !message.phases?.[selectedPhase]) return null;

  const phase = message.phases[selectedPhase];
  const details = phase.details;

  const getPhaseIcon = () => {
    switch (selectedPhase) {
      case "research":
        return <Search className="w-5 h-5 text-research" />;
      case "analysis":
        return <BarChart3 className="w-5 h-5 text-analysis" />;
      case "thinking":
        return <Brain className="w-5 h-5 text-thinking" />;
    }
  };

  const getPhaseColor = () => {
    switch (selectedPhase) {
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
    <Dialog open={!!selectedPhase} onOpenChange={() => onClose()}>
      <DialogContent className="max-w-2xl bg-card border-border/50">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            {getPhaseIcon()}
            <span className="capitalize">{selectedPhase} Phase Details</span>
            {phase.status === "complete" && (
              <Badge variant="outline" className={`text-${getPhaseColor()} border-${getPhaseColor()}/30 bg-${getPhaseColor()}-bg`}>
                <CheckCircle className="w-3 h-3 mr-1" />
                Complete
                {phase.duration && (
                  <>
                    <Clock className="w-3 h-3 ml-2 mr-1" />
                    {(phase.duration / 1000).toFixed(1)}s
                  </>
                )}
              </Badge>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {selectedPhase === "research" && details && "sources" in details && (
            <>
              <div>
                <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                  <Search className="w-4 h-4" />
                  Sources Consulted
                </h4>
                <ul className="space-y-2">
                  {details.sources.map((source, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <CheckCircle className="w-3 h-3 mt-0.5 text-research flex-shrink-0" />
                      {source}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-3">Search Queries</h4>
                <div className="space-y-2">
                  {details.queries.map((query, index) => (
                    <Badge key={index} variant="outline" className="mr-2 mb-2 text-xs bg-research-bg border-research/30">
                      {query}
                    </Badge>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-3">Key Findings</h4>
                <ul className="space-y-2">
                  {details.findings.map((finding, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                      <div className="w-1.5 h-1.5 bg-research rounded-full mt-2 flex-shrink-0" />
                      {finding}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}

          {selectedPhase === "analysis" && details && "dataPoints" in details && (
            <>
              <div>
                <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Data Points
                </h4>
                <ul className="space-y-2">
                  {details.dataPoints.map((point, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                      <CheckCircle className="w-3 h-3 mt-0.5 text-analysis flex-shrink-0" />
                      {point}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-3">Patterns Identified</h4>
                <ul className="space-y-2">
                  {details.patterns.map((pattern, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                      <div className="w-1.5 h-1.5 bg-analysis rounded-full mt-2 flex-shrink-0" />
                      {pattern}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-3">Strategic Insights</h4>
                <ul className="space-y-2">
                  {details.insights.map((insight, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                      <div className="w-1.5 h-1.5 bg-analysis rounded-full mt-2 flex-shrink-0" />
                      {insight}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}

          {selectedPhase === "thinking" && details && "reasoning" in details && (
            <>
              <div>
                <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                  <Brain className="w-4 h-4" />
                  Reasoning Process
                </h4>
                <ul className="space-y-2">
                  {details.reasoning.map((reason, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                      <CheckCircle className="w-3 h-3 mt-0.5 text-thinking flex-shrink-0" />
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-3">Key Considerations</h4>
                <ul className="space-y-2">
                  {details.considerations.map((consideration, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                      <div className="w-1.5 h-1.5 bg-thinking rounded-full mt-2 flex-shrink-0" />
                      {consideration}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-3">Final Conclusions</h4>
                <ul className="space-y-2">
                  {details.conclusions.map((conclusion, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                      <div className="w-1.5 h-1.5 bg-thinking rounded-full mt-2 flex-shrink-0" />
                      {conclusion}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};