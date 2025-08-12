# python -m unittest test/test_finetune_llama3.py

import unittest
import os
import json
import shutil
from unittest.mock import patch, MagicMock

# We need to import the script we are testing.
# To do this, we add its directory to the python path.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'rag')))

import finetune_llama3

class TestFinetuneLlama3(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and mock data for each test."""
        self.test_dir = "temp_test_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # Override the paths in the original script to use our temp directory
        self.original_chat_log_path = finetune_llama3.CHAT_LOG_PATH
        self.original_processed_log_path = finetune_llama3.PROCESSED_LOG_PATH
        self.original_output_dir = finetune_llama3.OUTPUT_DIR
        
        finetune_llama3.CHAT_LOG_PATH = os.path.join(self.test_dir, "chat_logs.jsonl")
        finetune_llama3.PROCESSED_LOG_PATH = os.path.join(self.test_dir, "chat_logs_processed.jsonl")
        finetune_llama3.OUTPUT_DIR = os.path.join(self.test_dir, "llama3-finetuned-lora")

        # Mock model name to avoid actual downloads if patches fail
        finetune_llama3.MODEL_NAME = "mock-model"

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.test_dir)
        # Restore original paths
        finetune_llama3.CHAT_LOG_PATH = self.original_chat_log_path
        finetune_llama3.PROCESSED_LOG_PATH = self.original_processed_log_path
        finetune_llama3.OUTPUT_DIR = self.original_output_dir

    def _create_log_file(self, lines):
        """Helper to create the chat log file with given lines."""
        with open(finetune_llama3.CHAT_LOG_PATH, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')

    @patch('finetune_llama3.SFTTrainer')
    @patch('finetune_llama3.AutoModelForCausalLM.from_pretrained')
    @patch('finetune_llama3.AutoTokenizer.from_pretrained')
    def test_pipeline_aborts_if_not_enough_data(self, mock_tokenizer, mock_model, mock_trainer):
        """
        Test that the main pipeline does not run if the number of 'up' feedback
        logs is below the FINETUNE_THRESHOLD.
        """
        # Arrange: Create a log file with fewer than the threshold number of good logs
        finetune_llama3.FINETUNE_THRESHOLD = 5
        logs = [{"query": f"q{i}", "answer": f"a{i}", "feedback": "up"} for i in range(4)]
        self._create_log_file(logs)

        # Act
        finetune_llama3.main()

        # Assert
        mock_model.assert_not_called()
        mock_tokenizer.assert_not_called()
        mock_trainer.assert_not_called()
        # Check that logs were not archived
        self.assertFalse(os.path.exists(finetune_llama3.PROCESSED_LOG_PATH))

    @patch('finetune_llama3.SFTTrainer')
    @patch('finetune_llama3.AutoModelForCausalLM.from_pretrained')
    @patch('finetune_llama3.AutoTokenizer.from_pretrained')
    def test_pipeline_runs_with_enough_data(self, mock_tokenizer, mock_model, mock_trainer):
        """
        Test the full pipeline runs correctly when there is enough data,
        and that logs are archived correctly.
        """
        # Arrange
        finetune_llama3.FINETUNE_THRESHOLD = 5
        good_logs = [{"query": f"q{i}", "answer": f"a{i}", "feedback": "up"} for i in range(5)]
        bad_logs = [{"query": "bad_q", "answer": "bad_a", "feedback": "down"}]
        malformed_line = "this is not json"
        
        # Create a mock trainer instance to check calls
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        with open(finetune_llama3.CHAT_LOG_PATH, 'w') as f:
            for log in good_logs + bad_logs:
                f.write(json.dumps(log) + '\n')
            f.write(malformed_line + '\n')

        # Act
        finetune_llama3.main()

        # Assert
        # 1. Check that model loading and training was called
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once()

        # 2. Check that the trainer was initialized with the correct number of samples
        _, kwargs = mock_trainer.call_args
        self.assertEqual(len(kwargs['train_dataset']), len(good_logs))

        # 3. Check that logs were archived correctly
        self.assertTrue(os.path.exists(finetune_llama3.PROCESSED_LOG_PATH))
        with open(finetune_llama3.PROCESSED_LOG_PATH, 'r') as f:
            processed_lines = f.readlines()
            self.assertEqual(len(processed_lines), len(good_logs))

        # 4. Check that the original log file now only contains the remaining logs
        with open(finetune_llama3.CHAT_LOG_PATH, 'r') as f:
            remaining_lines = f.readlines()
            self.assertEqual(len(remaining_lines), 2) # bad_log + malformed_line
            self.assertEqual(json.loads(remaining_lines[0]), bad_logs[0])
            self.assertEqual(remaining_lines[1].strip(), malformed_line)

    def test_load_new_qa_pairs_handles_edge_cases(self):
        """
        Test the data loading function with various edge cases.
        """
        # Case 1: File does not exist
        if os.path.exists(finetune_llama3.CHAT_LOG_PATH):
            os.remove(finetune_llama3.CHAT_LOG_PATH)
        pairs = finetune_llama3.load_new_qa_pairs()
        self.assertEqual(len(pairs), 0)

        # Case 2: Empty file
        self._create_log_file([])
        pairs = finetune_llama3.load_new_qa_pairs()
        self.assertEqual(len(pairs), 0)

        # Case 3: File with malformed JSON and mixed feedback
        logs = [
            {"query": "q1", "answer": "a1", "feedback": "up"},
            {"query": "q2", "answer": "a2", "feedback": "down"},
            {"query": "q3", "answer": "a3", "feedback": "up"},
        ]
        with open(finetune_llama3.CHAT_LOG_PATH, 'w') as f:
            f.write(json.dumps(logs[0]) + '\n')
            f.write("not a valid json\n")
            f.write(json.dumps(logs[1]) + '\n')
            f.write(json.dumps(logs[2]) + '\n')
        
        pairs = finetune_llama3.load_new_qa_pairs()
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]['input'], 'q1')
        self.assertEqual(pairs[1]['input'], 'q3')

if __name__ == '__main__':
    unittest.main()
