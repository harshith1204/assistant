import { ConversationalChat } from "@/components/ConversationalChat";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        <h1 className="text-4xl font-bold mb-8 text-center">AI Research & Business Assistant</h1>
        <div className="h-[calc(100vh-120px)]">
          <ConversationalChat />
        </div>
      </div>
    </div>
  );
};

export default Index;
