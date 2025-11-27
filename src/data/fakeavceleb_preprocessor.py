import os
import random
from typing import Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

from src.data.base_preprocessor import DataPreprocessor


class FakeAVCelebPreprocessor(DataPreprocessor):
    """Preprocessor for the FakeAVCeleb dataset."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the FakeAVCeleb preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        super().__init__(config)
        self.dataset_name = "FakeAVCeleb_v1.2"
        self.metadata_filename = "meta_data.csv"
        self.metadata = None
        self.category_mapping = {
            'A': 'RealVideo-RealAudio',
            'B': 'RealVideo-FakeAudio',
            'C': 'FakeVideo-RealAudio',
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
        row = self.metadata[self.metadata['filename'] == filename]
        if len(row) == 0:
            raise ValueError(f"No metadata found for video: {filename}")

        row = row.iloc[0]

        # Determine label (0 for real, 1 for fake)
        label = 0 if row['category'] == 'A' else 1

        # Create metadata dictionary
        metadata = {
            'source': row['source'],
            'target1': row['target1'],
            'target2': row['target2'],
            'method': row['method'],
            'category': row['category'],
            'type': row['type'],
            'gender': row['gender'],
            'race': row['race'],
            'filename': row['filename'],
            'path': row['path']
        }

        return label, metadata

    def process_dataset(self, input_dir: str, output_dir: str):
        """Process the FakeAVCeleb dataset.
        
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
            'methods': {},
            'gender_distribution': {},
            'race_distribution': {}
        }

        # Initialize shards-only storage
        self.sample_index = 0
        self._initialize_output_storage(output_dir)

        # 1. Collect all video files from all categories
        all_video_files = []
        print("Collecting video files...")
        
        for category in ['A', 'B', 'C', 'D']:
            category_dir = os.path.join(input_dir, self.dataset_name, self.category_mapping[category])
            if not os.path.exists(category_dir):
                continue

            # Get list of all video files in the category
            files_in_category = []
            for root, _, files in os.walk(category_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        files_in_category.append(os.path.join(root, file))
            
            # TODO: REMOVE THIS (Limit for testing/debugging)
            random.shuffle(files_in_category)
            files_in_category = files_in_category[:500]
            
            all_video_files.extend(files_in_category)

        # 2. Global Shuffle to ensure mixed shards
        print(f"Found {len(all_video_files)} total videos. Shuffling...")
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
        category = metadata['category']
        stats['categories'][category] = stats['categories'].get(category, 0) + 1

        # Update method statistics
        method = metadata['method']
        stats['methods'][method] = stats['methods'].get(method, 0) + 1

        # Update gender statistics
        gender = metadata['gender']
        stats['gender_distribution'][gender] = stats['gender_distribution'].get(gender, 0) + 1

        # Update race statistics
        race = metadata['race']
        stats['race_distribution'][race] = stats['race_distribution'].get(race, 0) + 1
