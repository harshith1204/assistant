import { ConversationalChat } from "@/components/ConversationalChat";
import { useParams } from "react-router-dom";

const Index = () => {
  const { userId } = useParams();
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        <h1 className="text-4xl font-bold mb-8 text-center">AI Research & Business Assistant</h1>
        <div className="h-[calc(100vh-120px)]">
          <ConversationalChat userId={userId} />
        </div>
      </div>
    </div>
  );
};

export default Index;
