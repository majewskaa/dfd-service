import os
import glob
import pandas as pd
from typing import Dict, Any, Tuple
from tqdm import tqdm

from src.data.base_preprocessor import DataPreprocessor

class PretrainPreprocessor(DataPreprocessor):
    """Preprocessor for the pretraining dataset (unlabeled videos)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_name = "pretrain"
        self.metadata_filename = "meta_data.csv"
        self.metadata = None

    def generate_metadata(self, input_dir: str) -> pd.DataFrame:
        """Generate metadata for the pretrain dataset.
        
        Args:
            input_dir: Directory containing the dataset
            
        Returns:
            DataFrame containing metadata
        """
        print(f"Scanning for videos in {input_dir}...")
        # Recursive search for mp4 files
        video_files = glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)
        
        data = []
        for file_path in video_files:
            filename = os.path.basename(file_path)
            # Relative path from input_dir
            rel_path = os.path.relpath(file_path, input_dir)
            
            data.append({
                'filename': filename,
                'path': rel_path,
                'label': 0, # Dummy label for pretraining
                'split': 'train' # All data is for training
            })
            
        df = pd.DataFrame(data)
        print(f"Found {len(df)} videos.")
        return df

    def load_metadata(self, input_dir: str):
        """Load or generate dataset metadata.
        
        Args:
            input_dir: Directory containing the dataset
        """
        metadata_path = os.path.join(input_dir, self.metadata_filename)
        
        if os.path.exists(metadata_path):
            print(f"Loading existing metadata from {metadata_path}")
            self.metadata = pd.read_csv(metadata_path)
        else:
            print("Generating new metadata...")
            self.metadata = self.generate_metadata(input_dir)
            self.metadata.to_csv(metadata_path, index=False)
            print(f"Saved metadata to {metadata_path}")

    def get_video_label(self, video_path: str) -> Tuple[int, Dict[str, Any]]:
        """Get label and metadata for a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (label, metadata)
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")

        filename = os.path.basename(video_path)
        
        # Simple lookup
        row = self.metadata[self.metadata['filename'] == filename]
        if len(row) == 0:
            # Fallback if filename not unique or not found (shouldn't happen if generated fresh)
            raise ValueError(f"No metadata found for video: {filename}")

        row = row.iloc[0]
        
        label = int(row['label'])
        
        metadata = {
            'filename': row['filename'],
            'path': row['path'],
            'split': row.get('split', 'train')
        }

        return label, metadata

    def process_dataset(self, input_dir: str, output_dir: str):
        """Process the pretrain dataset.
        
        Args:
            input_dir: Directory containing the dataset
            output_dir: Directory to save processed data
        """
        dataset_input_dir = os.path.join(input_dir, self.dataset_name)
        
        # Load or generate metadata
        self.load_metadata(dataset_input_dir)

        # Create output directory
        output_dir = os.path.join(output_dir, self.dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize statistics tracking
        stats = {
            'total_samples': 0,
            'successful_samples': 0,
            'failed_samples': 0
        }

        # Initialize shards-only storage
        self.sample_index = 0
        self._initialize_output_storage(output_dir)

        import random
        
        # Get list of videos from metadata
        # Construct full paths
        video_paths = [os.path.join(dataset_input_dir, row['path']) for _, row in self.metadata.iterrows()]

        # Shuffle and limit to 2500
        print(f"Found {len(video_paths)} videos. Shuffling and limiting to 2500...")
        random.shuffle(video_paths)
        video_paths = video_paths[:2500]

        print(f"Processing {len(video_paths)} videos...")
        
        for video_path in tqdm(video_paths, desc="Processing Dataset", unit="video"):
            try:
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found: {video_path}")
                    stats['failed_samples'] += 1
                    continue

                label, metadata = self.get_video_label(video_path)

                # Process video
                result = self.process_video(video_path, label)
                
                if result is not None:
                    # Add metadata to result
                    result['metadata'].update(metadata)

                    # Save result incrementally
                    self._save_incremental(result, output_dir)
                    
                    stats['successful_samples'] += 1
                else:
                    stats['failed_samples'] += 1
                
                stats['total_samples'] += 1

            except Exception as e:
                print(f"\nError processing {video_path}: {str(e)}")
                stats['failed_samples'] += 1
                continue

        self._finalize_output_storage(output_dir)
        self.save_dataset_statistics(stats, output_dir)
