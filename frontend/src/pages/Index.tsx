import { ConversationalChat } from "@/components/ConversationalChat";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChatInterfaceReal } from "@/components/ChatInterfaceReal";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        <h1 className="text-4xl font-bold mb-8 text-center">AI Research & Business Assistant</h1>
        
        <Tabs defaultValue="conversational" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="conversational">Conversational AI</TabsTrigger>
            <TabsTrigger value="classic">Classic Interface</TabsTrigger>
          </TabsList>
          
          <TabsContent value="conversational" className="h-[calc(100vh-200px)]">
            <ConversationalChat />
          </TabsContent>
          
          <TabsContent value="classic" className="h-[calc(100vh-200px)]">
            <ChatInterfaceReal />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
