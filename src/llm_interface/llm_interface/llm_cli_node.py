#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from llm_msgs.srv import Chat


class LLMCLINode(Node):
    def __init__(self):
        super().__init__("llm_cli_node")

        self.client = self.create_client(Chat, "/llm/chat")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /llm/chat service...")

        self.session_id = "terminal"
        self.get_logger().info(f"[LLM CLI] session={self.session_id}")
        self.get_logger().info("Type text and press Enter.")
        self.get_logger().info("Commands: /reset, /session <name>, /exit")

    def call_llm(self, text: str, reset: bool = False) -> str:
        req = Chat.Request()
        req.session_id = self.session_id
        req.user_text = text
        req.reset = reset

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            return "[ERROR] No response from LLM service"

        return future.result().assistant_text


def main():
    rclpy.init()
    node = LLMCLINode()

    try:
        while rclpy.ok():
            user_input = input("> ").strip()
            if not user_input:
                continue

            # ---------- CLI-only commands ----------
            if user_input == "/exit":
                print("[LLM CLI] exit")
                break

            if user_input == "/reset":
                node.call_llm("", reset=True)
                print("[LLM CLI] session reset")
                continue

            if user_input.startswith("/session"):
                parts = user_input.split(maxsplit=1)
                if len(parts) != 2:
                    print("[LLM CLI] usage: /session <name>")
                    continue
                node.session_id = parts[1]
                print(f"[LLM CLI] switched to session={node.session_id}")
                continue

            # ---------- normal chat ----------
            reply = node.call_llm(user_input)
            #print(f"[DEBUG] raw reply: {repr(reply)}")

            # ---- lock-aware behavior ----
            if "LLM is locked" in reply or "locked" in reply.lower():
                print("[LLM CLI]: Waiting for face authentication, no access granted")
                continue

            print(f"LLM: {reply}")

    except KeyboardInterrupt:
        print("\n[LLM CLI] interrupted")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
