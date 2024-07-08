import unittest
from src.agent_framework.perception_agent import PerceptionAgent

class TestPerceptionAgent(unittest.TestCase):
    def setUp(self):
        self.agent = PerceptionAgent('models/compressed/quantized_vision_model.pth', 'configs/agent_config.yaml')

    def test_perceive(self):
        result = self.agent.perceive('data/test/test_image.jpg')
        self.assertIn('scene_type', result)
        self.assertIn('objects', result)
        self.assertIsInstance(result['objects'], list)

    def test_classify_scene(self):
        # This test assumes you have a test image and know its expected classification
        output, _ = self.agent.process_image('data/test/test_image.jpg')
        scene_type = self.agent.classify_scene(output)
        self.assertIn(scene_type, self.agent.config['scene_classes'])

    def test_detect_objects(self):
        _, image = self.agent.process_image('data/test/test_image.jpg')
        objects = self.agent.detect_objects(image)
        self.assertIsInstance(objects, list)
        if objects:
            self.assertIn('class', objects[0])
            self.assertIn('confidence', objects[0])
            self.assertIn('box', objects[0])
            self.assertIn('distance', objects[0])

if __name__ == '__main__':
    unittest.main()