"use client";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, X, Send, User, Bot, ThumbsUp, ThumbsDown } from "lucide-react";

// --- Shadcn UI Components (assuming they are in these paths) ---
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "bot",
      content:
        "Hello! How can I help you with Panjab University admissions today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState([]);

  const messagesEndRef = useRef(null);
  const chatContentRef = useRef(null);

  // Auto-scrolling to the bottom of the chat
  useEffect(() => {
    if (chatContentRef.current) {
      chatContentRef.current.scrollTop = chatContentRef.current.scrollHeight;
    }
  }, [messages]);

  async function sendMessage() {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input, history: messages }),
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { role: "bot", content: data.answer }]);
    } catch (error) {
      console.error("Failed to send message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: "Sorry, something went wrong. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleFeedback(query, answer, feedback, index) {
    try {
      await fetch("http://localhost:8000/api/log_chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, answer, feedback }),
      });
      setFeedbackSent((prev) => [...prev, index]);
    } catch (error) {
      console.error("Failed to send feedback:", error);
    }
  }

  return (
    <>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="fixed bottom-20 right-6 w-80 h-[500px] bg-white shadow-xl rounded-lg flex flex-col border overflow-hidden"
          >
            <header className="bg-gray-50 p-3 border-b flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2">
                <Bot className="w-5 h-5 text-gray-600" />
                <h2 className="font-semibold text-gray-800">Admissions Chatbot</h2>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)} className="text-gray-500 hover:bg-gray-100">
                <X className="w-4 h-4" />
              </Button>
            </header>

            <div ref={chatContentRef} className="flex-1 p-3 overflow-y-auto space-y-3">
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2, delay: i * 0.05 }}
                  className={`flex items-start gap-2 ${
                    msg.role === "user" ? "justify-end" : ""
                  }`}
                >
                  {msg.role === "bot" && (
                    <Avatar className="w-7 h-7 bg-gray-200">
                      <AvatarFallback>B</AvatarFallback>
                    </Avatar>
                  )}
                  <div
                    className={`max-w-[70%] px-3 py-2 rounded-lg ${
                      msg.role === "user"
                        ? "bg-blue-500 text-white rounded-br-none"
                        : "bg-gray-200 text-gray-800 rounded-bl-none"
                    }`}
                  >
                    <p className="text-sm leading-snug">{msg.content}</p>
                  </div>
                  {msg.role === "user" && (
                    <Avatar className="w-7 h-7 bg-blue-100">
                      <AvatarFallback>U</AvatarFallback>
                    </Avatar>
                  )}
                  {msg.role === "bot" && i > 0 && (
                    <div className="flex justify-start items-center gap-1 mt-1 ml-9">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="w-6 h-6 text-gray-400 hover:text-green-500"
                        onClick={() => handleFeedback(messages[i-1]?.content, msg.content, "up", i)}
                        disabled={feedbackSent.includes(i)}
                      >
                        <ThumbsUp className={`w-4 h-4 ${feedbackSent.includes(i) ? 'text-green-500' : ''}`} />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="w-6 h-6 text-gray-400 hover:text-red-500"
                        onClick={() => handleFeedback(messages[i-1]?.content, msg.content, "down", i)}
                        disabled={feedbackSent.includes(i)}
                      >
                        <ThumbsDown className={`w-4 h-4 ${feedbackSent.includes(i) ? 'text-red-500' : ''}`} />
                      </Button>
                    </div>
                  )}
                </motion.div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <div className="p-3 border-t flex shrink-0">
              <Input
                className="flex-1 border rounded-lg p-2 text-sm"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !isLoading && sendMessage()}
                disabled={isLoading}
                placeholder="Type your message..."
              />
              <Button
                onClick={sendMessage}
                disabled={isLoading}
                className="ml-2 bg-blue-500 text-white px-3 py-2 rounded-lg"
              >
                <Send size={18} />
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        className="fixed bottom-6 right-6 bg-blue-600 p-4 rounded-full shadow-lg text-white flex items-center justify-center"
        onClick={() => setIsOpen(!isOpen)}
      >
        <AnimatePresence initial={false} mode="wait">
          <motion.div
            key={isOpen ? "x" : "message"}
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 20, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {isOpen ? <X size={28} /> : <MessageCircle size={28} />}
          </motion.div>
        </AnimatePresence>
      </motion.button>
    </>
  );
}
