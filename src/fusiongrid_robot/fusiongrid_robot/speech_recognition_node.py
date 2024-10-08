import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from audio_common_msgs.msg import AudioStamped
from std_msgs.msg import String

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import noisereduce as nr
#import pyaudio
from audio_common.utils import msg_to_array

class SpeechRecognitionNode(Node):

    def __init__(self):
        super().__init__('speech_recognition_node')

        # Noise detection parameters
        self.noise_threshold = 40 # Adjust as needed based on environment
        self.silence_duration = 2.0  # Seconds of silence to consider speech finished
        self.silence_start_time = None
        self.is_collecting_audio = False
        self.audio_buffer = []
        self.buffer_duration = 5.0  # Default buffer duration
        self.sample_rate = 16000

        # Load Whisper model with chunked inference
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_id = "openai/whisper-large-v3-turbo"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device_str)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            #chunk_length_s=10,  # Enable chunked mode with 30-second chunks
            batch_size=16,       # Batch size for inference, adjust based on your device
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=self.device_str,
            generate_kwargs={"language": "english"}
        )

        # Subscription and publisher
        self.subscription = self.create_subscription(
            AudioStamped,
            'audio',
            self.audio_callback,
            qos_profile_sensor_data
        )
        self.text_publisher = self.create_publisher(String, 'speech_text', 10)
        self.get_logger().info("SpeechRecognitionNode initialized")

    def audio_callback(self, msg: AudioStamped):
        array_data = msg_to_array(msg.audio)
        if array_data is None:
            self.get_logger().error(f"Format {msg.audio.info.format} unknown")
            return

        # Check the noise level in the current audio data
        array_data = self.reduce_noise(array_data, self.sample_rate)
        rms = np.sqrt(np.mean(np.abs(array_data)))
        self.get_logger().info(f"Noise level (RMS): {rms}")

        if rms > self.noise_threshold:
            self.get_logger().info("Speech detected, collecting audio...")
            self.is_collecting_audio = True
            self.silence_start_time = None
            self.audio_buffer.append(array_data)
        else:
            if self.is_collecting_audio:
                if self.silence_start_time is None:
                    self.silence_start_time = self.get_clock().now().to_msg().sec
                elapsed_silence_time = self.get_clock().now().to_msg().sec - self.silence_start_time

                # Stop collecting audio if silence duration exceeds threshold
                if elapsed_silence_time >= self.silence_duration:
                    self.get_logger().info("Silence detected, processing collected audio...")
                    self.process_buffered_audio()
                    self.is_collecting_audio = False
                    self.audio_buffer = []  # Clear the buffer

    def process_buffered_audio(self):
        if not self.audio_buffer:
            return

        concatenated_audio = np.concatenate(self.audio_buffer)
        transcription = self.process_speech(concatenated_audio, self.sample_rate)

        if transcription:
            text_msg = String()
            text_msg.data = transcription
            self.text_publisher.publish(text_msg)
            self.get_logger().info(f"Transcription published: {transcription}")


    def reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        # Perform noise reduction on the audio
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
        return reduced_noise_audio


    def process_speech(self, audio_data: np.ndarray, sample_rate: int) -> str:
        inputs = {
            "array": audio_data,
            "sampling_rate": sample_rate
        }
        result = self.pipe(inputs)
        return result.get("text", "")

    def destroy_node(self) -> bool:
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
