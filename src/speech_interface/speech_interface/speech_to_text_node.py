# speech_interface/speech_to_text_node.py

import rclpy
from rclpy.node import Node

import speech_recognition as sr


class SpeechToTextNode(Node):
    def __init__(self):
        super().__init__('speech_to_text_node')

        # 创建 SpeechRecognition 的对象
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.get_logger().info('SpeechToTextNode 初始化完成。')
        self.get_logger().info('每轮会让你说一句话，然后在终端打印识别结果。')

    def listen_and_recognize(self):
        """从麦克风录一段音，并转成文字打印到终端"""
        try:
            with self.microphone as source:
                self.get_logger().info('请说话（保持安静环境，Ctrl+C 退出）...')
                # 根据环境噪声自动调整
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # 监听一段语音，这里限制最长 5 秒
                audio = self.recognizer.listen(source, phrase_time_limit=5)

            try:
                # 使用 Google 的在线识别服务（需要能上网）
                text = self.recognizer.recognize_google(audio, language='zh-CN')
                self.get_logger().info(f'识别结果: {text}')
                print(f'[SpeechToText] {text}', flush=True)
            except sr.UnknownValueError:
                self.get_logger().warn('没听清你在说什么（UnknownValueError）。')
            except sr.RequestError as e:
                self.get_logger().error(f'语音识别服务出错: {e}')

        except Exception as e:
            self.get_logger().error(f'录音或识别过程中发生异常: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SpeechToTextNode()

    try:
        # 简单暴力：循环调用 listen_and_recognize
        while rclpy.ok():
            node.listen_and_recognize()
            # 让 ROS 事件循环跑一下（这里不订阅任何话题，只是规范一点）
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('SpeechToTextNode 关闭。')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
