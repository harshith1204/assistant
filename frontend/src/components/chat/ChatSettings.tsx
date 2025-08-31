import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { X, Search, Zap, Brain, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatSettingsProps {
  researchEnabled: boolean;
  setResearchEnabled: (enabled: boolean) => void;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  onClose: () => void;
}

const models = [
  { id: "gpt-4", name: "GPT-4", icon: Brain, description: "Most capable model" },
  { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo", icon: Zap, description: "Fast and efficient" },
  { id: "claude-3", name: "Claude 3", icon: Cpu, description: "Advanced reasoning" },
  { id: "gemini-pro", name: "Gemini Pro", icon: Brain, description: "Google's latest" },
];

export const ChatSettings = ({ 
  researchEnabled, 
  setResearchEnabled, 
  selectedModel, 
  setSelectedModel, 
  onClose 
}: ChatSettingsProps) => {
  return (
    <Card className="mx-4 mb-4 bg-card/80 backdrop-blur-sm border-border/50">
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">Settings</h3>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>
        
        <div className="space-y-6">
          {/* Research Toggle */}
          <div className="space-y-2">
            <Label className="text-sm font-medium text-foreground">Research Mode</Label>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Search className="w-4 h-4 text-research" />
                <div>
                  <p className="text-sm text-foreground">Enable Research Phase</p>
                  <p className="text-xs text-muted-foreground">
                    Gather additional context before responding
                  </p>
                </div>
              </div>
              <Switch
                checked={researchEnabled}
                onCheckedChange={setResearchEnabled}
                className="data-[state=checked]:bg-research"
              />
            </div>
          </div>

          {/* Model Selection */}
          <div className="space-y-2">
            <Label className="text-sm font-medium text-foreground">AI Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="bg-background/50 border-border/50">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent className="bg-popover border-border/50">
                {models.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div className="flex items-center gap-2">
                      <model.icon className="w-4 h-4" />
                      <div>
                        <p className="font-medium">{model.name}</p>
                        <p className="text-xs text-muted-foreground">{model.description}</p>
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Performance Info */}
          <div className="p-3 rounded-lg bg-gradient-primary border border-border/30">
            <div className="flex items-start gap-2">
              <Zap className="w-4 h-4 text-analysis mt-0.5" />
              <div>
                <p className="text-sm font-medium text-foreground">Performance Tips</p>
                <ul className="text-xs text-muted-foreground mt-1 space-y-1">
                  <li>• Research mode adds 2-3s but provides better context</li>
                  <li>• GPT-4 is slower but more accurate than GPT-3.5</li>
                  <li>• All responses include analysis and thinking phases</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};