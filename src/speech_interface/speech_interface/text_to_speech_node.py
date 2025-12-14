# speech_interface/text_to_speech_node.py

import rclpy
from rclpy.node import Node

import pyttsx3


class TextToSpeechNode(Node):
    def __init__(self):
        super().__init__('text_to_speech_node')

        # 初始化 TTS 引擎
        self.engine = pyttsx3.init()
        # 可选：调节语速和音量
        self.engine.setProperty('rate', 180)   # 语速
        self.engine.setProperty('volume', 1.0) # 音量 0.0 ~ 1.0

        self.get_logger().info('TextToSpeechNode 初始化完成。')
        self.get_logger().info('在终端输入一行文字按回车，我会帮你朗读。输入 q/quit/exit 退出。')

    def speak(self, text: str):
        text = text.strip()
        if not text:
            self.get_logger().warn('输入为空，不朗读。')
            return

        self.get_logger().info(f'朗读内容: {text}')
        self.engine.say(text)
        self.engine.runAndWait()


def main(args=None):
    rclpy.init(args=args)
    node = TextToSpeechNode()

    try:
        while rclpy.ok():
            # 从终端读一行输入
            user_input = input('请输入要朗读的文本（q/quit/exit 退出）： ')
            if user_input.strip().lower() in ['q', 'quit', 'exit']:
                break

            node.speak(user_input)
            # 同样让 ROS 跑一下
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('TextToSpeechNode 关闭。')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
