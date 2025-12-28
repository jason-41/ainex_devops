#!/usr/bin/env python3
#modify the source path to venv path
import rclpy
from rclpy.node import Node
from llm_msgs.srv import Chat
from llm_interface.openai_client import OpenAIChatClient
from pathlib import Path

#from openai import OpenAI
#!/home/wbo/hrs_vision_env/bin/python3

class LLMServerNode(Node):
    def __init__(self):
        super().__init__("llm_server_node")
        self.srv = self.create_service(Chat, "/llm/chat", self.on_chat)
        self.get_logger().info("LLM Server ready at /llm/chat")

        # memory：session_id -> list[str]
        self.memory = {}
        self.client = OpenAIChatClient(model="gpt-4.1-mini")

        # load system prompt from file
        prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt not found: {prompt_path}")

        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()

        self.get_logger().info("System prompt loaded successfully")



    def on_chat(self, req: Chat.Request, res: Chat.Response):
        sid = req.session_id.strip() or "default"
        if req.reset or sid not in self.memory:
            self.memory[sid] = []

        # build messages（system + history + user）
        messages = []

        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        for m in self.memory[sid]:
            messages.append(m)

        messages.append({
            "role": "user",
            "content": req.user_text
        })

        try:
            reply = self.client.chat(messages)
        except Exception as e:
            self.get_logger().error(f"OpenAI API error: {e}")
            reply = f"[ERROR] {e}"

        # write back to memory
        self.memory[sid].append({"role": "user", "content": req.user_text})
        self.memory[sid].append({"role": "assistant", "content": reply})

        res.assistant_text = reply

        return res

def main():
    rclpy.init()
    node = LLMServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
