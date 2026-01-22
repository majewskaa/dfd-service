import os
import random
from typing import Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

from src.data.base_preprocessor import DataPreprocessor


class DeepfakeEval2024Preprocessor(DataPreprocessor):
    """Preprocessor for the Deepfake Eval 2024 dataset."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Deepfake Eval 2024 preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        super().__init__(config)
        self.dataset_name = "Deepfake_Eval_2024"
        self.metadata_filename = "meta_data.csv"
        self.metadata = None
        self.category_mapping = {
            'A': 'RealVideo-RealAudio',
            'B': 'FakeVideo-RealAudio',
            'C': 'RealVideo-FakeAudio',
            'D': 'FakeVideo-FakeAudio'
        }

    def load_metadata(self, metadata_path: str):
        """Load dataset metadata from CSV file.
        
        Args:
            metadata_path: Path to the metadata CSV file
        """
        self.metadata = pd.read_csv(metadata_path)

    def get_video_label(self, video_path: str) -> Tuple[int, Dict[str, Any]]:
        """Get label and metadata for a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (label, metadata)
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")

        # Extract filename from path
        filename = os.path.basename(video_path)

        # Find matching row in metadata
        row = self.metadata[self.metadata['Filename'] == filename]
        if len(row) == 0:
            # Try matching without extension if needed, or just fail
            # Some datasets might have inconsistencies
            raise ValueError(f"No metadata found for video: {filename}")

        row = row.iloc[0]

        # Determine label (0 for real, 1 for fake)
        # Category A is Real, others are Fake
        label = 0 if row['category'] == 'A' else 1

        # Create metadata dictionary
        # We include available fields from the CSV
        metadata = {
            'filename': row['Filename'],
            'date': row.get('Date', ''),
            'video_ground_truth': row.get('Video Ground Truth', ''),
            'audio_ground_truth': row.get('Audio Ground Truth', ''),
            'category': row['category'],
            'path': video_path
        }

        return label, metadata

    def process_dataset(self, input_dir: str, output_dir: str):
        """Process the Deepfake Eval 2024 dataset.
        
        Args:
            input_dir: Directory containing the dataset
            output_dir: Directory to save processed data
        """
        # Load metadata
        print("Loading metadata...")
        metadata_path = os.path.join(input_dir, self.dataset_name, self.metadata_filename)
        self.load_metadata(metadata_path)

        # Create output directory
        output_dir = os.path.join(output_dir, self.dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize statistics tracking
        stats = {
            'total_samples': 0,
            'categories': {},
            'video_ground_truth': {},
            'audio_ground_truth': {}
        }

        # Initialize shards-only storage
        self.sample_index = 0
        self._initialize_output_storage(output_dir)

        # 1. Collect all video files
        print("Collecting video files...")
        video_data_dir = os.path.join(input_dir, self.dataset_name, "video-data")
        
        if not os.path.exists(video_data_dir):
             raise FileNotFoundError(f"Video data directory not found: {video_data_dir}")

        all_video_files = []
        
        # Iterate through metadata to find files, ensuring we only process files we have metadata for
        # and that exist on disk
        files_by_category = {}
        
        for _, row in self.metadata.iterrows():
            filename = row['Filename']
            category = row['category']
            file_path = os.path.join(video_data_dir, filename)
            
            if os.path.exists(file_path):
                if category not in files_by_category:
                    files_by_category[category] = []
                files_by_category[category].append(file_path)

        for category, files in files_by_category.items():
            limit = self.config.get("debug", {}).get("limit_per_category")
            if limit is not None and limit > 0:
                selected_files = files[:limit]
            else:
                selected_files = files
            
            all_video_files.extend(selected_files)
            print(f"Category {category}: Selected {len(selected_files)}/{len(files)} videos")

        # 2. Global Shuffle to ensure mixed shards
        print(f"Found {len(all_video_files)} total videos matching metadata. Shuffling...")
        random.shuffle(all_video_files)
        
        # 3. Process all videos
        for video_path in tqdm(all_video_files, desc="Processing Dataset", unit="video"):
            try:
                # Get label and metadata
                label, metadata = self.get_video_label(video_path)

                # Process video
                result = self.process_video(video_path, label)
                if result is not None:
                    # Add metadata to result
                    result['metadata'].update(metadata)

                    # Save result incrementally
                    self._save_incremental(result, output_dir)

                    # Update statistics
                    self._update_statistics(stats, result['metadata'])
                    stats['total_samples'] += 1

            except Exception as e:
                # Log and continue with next video
                print(f"\nError processing {video_path}: {str(e)}")
                continue

        self._finalize_output_storage(output_dir)
        self.save_dataset_statistics(stats, output_dir)

    def _update_statistics(self, stats: Dict[str, Any], metadata: Dict[str, Any]):
        """Update statistics with metadata from a processed sample.
        
        Args:
            stats: Current statistics dictionary
            metadata: Metadata from processed sample
        """
        # Update category statistics
        category = metadata.get('category', 'Unknown')
        stats['categories'][category] = stats['categories'].get(category, 0) + 1

        # Update video ground truth statistics
        vgt = metadata.get('video_ground_truth', 'Unknown')
        stats['video_ground_truth'][vgt] = stats['video_ground_truth'].get(vgt, 0) + 1
        
        # Update audio ground truth statistics
        agt = metadata.get('audio_ground_truth', 'Unknown')
        stats['audio_ground_truth'][agt] = stats['audio_ground_truth'].get(agt, 0) + 1
