import os
import shutil
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import logging
from typing import Tuple, List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, expert_properties_path: str, output_dir: str, skip_augmentation: bool = False):
        """
        Initialize the data preparation process.
        
        Args:
            expert_properties_path (str): Path to the expert_properties folder
            output_dir (str): Path where processed data will be saved
            skip_augmentation (bool): If True, skip the data augmentation step
        """
        self.expert_properties_path = expert_properties_path
        self.output_dir = output_dir
        self.required_folders = ['anchor', 'positive', 'negative']
        self.skip_augmentation = skip_augmentation
        
        # Initialize counters for dataset statistics
        self.total_subjects = 0
        self.total_anchor_images = 0
        self.total_positive_images = 0
        self.total_negative_images = 0
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

    def create_directory(self, directory: str) -> None:
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    def is_valid_subject_folder(self, folder_path: str) -> Tuple[bool, str]:
        """
        Check if the folder has all required subfolders and contains images.
        
        Args:
            folder_path (str): Path to the subject folder
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        # Check if all required folders exist
        missing_folders = []
        for subfolder in self.required_folders:
            if not os.path.isdir(os.path.join(folder_path, subfolder)):
                missing_folders.append(subfolder)
        
        if missing_folders:
            return False, f"Missing folders: {', '.join(missing_folders)}"
        
        # Check if each folder has at least one image
        empty_folders = []
        for subfolder in self.required_folders:
            subfolder_path = os.path.join(folder_path, subfolder)
            image_count = len([f for f in os.listdir(subfolder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if image_count == 0:
                empty_folders.append(subfolder)
        
        if empty_folders:
            return False, f"Empty folders (no images): {', '.join(empty_folders)}"
        
        return True, "Valid"

    def copy_folder_structure(self, source_path: str, target_path: str) -> str:
        """
        Copy the entire folder structure while maintaining the subject name.
        
        Args:
            source_path (str): Path to source subject folder
            target_path (str): Path to target directory
            
        Returns:
            str: Path to the created subject folder
        """
        # Get the subject folder name
        subject_folder = os.path.basename(source_path)
        target_subject_path = os.path.join(target_path, subject_folder)
        self.create_directory(target_subject_path)
        
        # Copy each subfolder
        for subdir in self.required_folders:
            source_subdir = os.path.join(source_path, subdir)
            target_subdir = os.path.join(target_subject_path, subdir)
            
            if os.path.exists(source_subdir):
                self.create_directory(target_subdir)
                
                # Copy all images
                count = 0
                for filename in os.listdir(source_subdir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        source_file = os.path.join(source_subdir, filename)
                        target_file = os.path.join(target_subdir, filename)
                        shutil.copy2(source_file, target_file)
                        count += 1
                logger.info(f"Copied {count} images to {target_subdir}")
        
        return target_subject_path

    def augment_images(self, folder_path: str, target_count: int) -> None:
        """
        Augment images in the folder until reaching target count.
        
        Args:
            folder_path (str): Path to the folder containing images
            target_count (int): Target number of images after augmentation
        """
        current_count = len([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if current_count >= target_count:
            return
        
        # Define augmentations
        augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.RandomRotation(20),           # Rotate by up to 20 degrees
            transforms.Compose([                     # Both flip and rotate
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(20)
            ])
        ]
        
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        aug_idx = 0
        pbar = tqdm(total=target_count - current_count, 
                    desc=f"Augmenting {os.path.basename(folder_path)}")
        
        while current_count < target_count:
            # Select random image and augmentation
            img_name = random.choice(images)
            aug = random.choice(augmentations)
            
            # Load and augment image
            img_path = os.path.join(folder_path, img_name)
            image = Image.open(img_path).convert('RGB')
            augmented_image = aug(image)
            
            # Save augmented image
            base_name, ext = os.path.splitext(img_name)
            new_name = f"aug_{aug_idx}_{base_name}{ext}"
            save_path = os.path.join(folder_path, new_name)
            augmented_image.save(save_path)
            
            current_count += 1
            aug_idx += 1
            pbar.update(1)
        
        pbar.close()

    def process_subject_folder(self, source_path: str) -> None:
        """
        Process a single subject folder.
        
        Args:
            source_path (str): Path to the source subject folder
        """
        # First, copy the entire structure and files
        target_subject_path = self.copy_folder_structure(source_path, self.output_dir)
        
        # Count images in each folder
        counts = {}
        for subdir in self.required_folders:
            folder_path = os.path.join(target_subject_path, subdir)
            if os.path.exists(folder_path):
                counts[subdir] = len([f for f in os.listdir(folder_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # If not skipping augmentation, balance the folders
        if not self.skip_augmentation:
            # Find maximum count for balancing
            max_count = max(counts.values())
            logger.info(f"\nBalancing folders to {max_count} images each...")
            
            # Augment each folder to reach the maximum count
            for subdir in self.required_folders:
                folder_path = os.path.join(target_subject_path, subdir)
                if os.path.exists(folder_path):
                    self.augment_images(folder_path, max_count)
        else:
            logger.info("\nSkipping data augmentation as requested")
        
        # Print final statistics and update counters
        logger.info(f"\nFinal image counts for {os.path.basename(target_subject_path)}:")
        for subdir in self.required_folders:
            folder_path = os.path.join(target_subject_path, subdir)
            if os.path.exists(folder_path):
                final_count = len([f for f in os.listdir(folder_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                logger.info(f"{subdir}: {final_count} images")
                
                # Update total counts
                if subdir == 'anchor':
                    self.total_anchor_images += final_count
                elif subdir == 'positive':
                    self.total_positive_images += final_count
                elif subdir == 'negative':
                    self.total_negative_images += final_count
        
        # Increment subject counter
        self.total_subjects += 1

    def prepare_data(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Main method to prepare the data.
        
        Returns:
            Tuple[List[str], List[Tuple[str, str]]]: (processed_folders, skipped_folders)
        """
        if not os.path.exists(self.expert_properties_path):
            raise ValueError("expert_properties path does not exist!")
        
        # Get all subject folders
        subject_folders = []
        skipped_folders = []
        
        logger.info("\nAnalyzing subject folders...")
        total_folders = len([d for d in os.listdir(self.expert_properties_path) 
                           if os.path.isdir(os.path.join(self.expert_properties_path, d))])
        
        for item in os.listdir(self.expert_properties_path):
            item_path = os.path.join(self.expert_properties_path, item)
            if os.path.isdir(item_path):
                is_valid, reason = self.is_valid_subject_folder(item_path)
                if is_valid:
                    subject_folders.append(item_path)
                else:
                    skipped_folders.append((item, reason))
        
        if not subject_folders:
            logger.warning("\nNo valid subject folders found!")
            return [], skipped_folders
        
        # Print summary
        logger.info(f"\nFolder Analysis Summary:")
        logger.info(f"Total folders found: {total_folders}")
        logger.info(f"Valid folders: {len(subject_folders)}")
        logger.info(f"Invalid folders: {len(skipped_folders)}")
        
        logger.info(f"\nValid subject folders to process:")
        for folder in subject_folders:
            logger.info(f"✓ {os.path.basename(folder)}")
        
        if skipped_folders:
            logger.warning(f"\nSkipped folders and reasons:")
            for folder_name, reason in skipped_folders:
                logger.warning(f"✗ {folder_name} - {reason}")
        
        # Process each valid subject folder
        logger.info("\nStarting processing...")
        for i, folder in enumerate(subject_folders, 1):
            logger.info(f"\nProcessing folder {i} of {len(subject_folders)}: {os.path.basename(folder)}")
            self.process_subject_folder(folder)
        
        # Log total dataset statistics
        logger.info("\n" + "="*50)
        logger.info("FINAL DATASET STATISTICS:")
        logger.info(f"Total subjects processed: {self.total_subjects}")
        logger.info(f"Total anchor images: {self.total_anchor_images}")
        logger.info(f"Total positive images: {self.total_positive_images}")
        logger.info(f"Total negative images: {self.total_negative_images}")
        logger.info(f"Total images in dataset: {self.total_anchor_images + self.total_positive_images + self.total_negative_images}")
        logger.info("="*50)
        
        return subject_folders, skipped_folders

def main():
    # Get paths from user
    expert_properties_path = input("Enter path to raw data: ")
    output_dir = input("Enter the path to destination dataset folder (or press Enter for current directory): ").strip()
    
    # Ask about skipping augmentation
    skip_augmentation = input("Skip data augmentation? (y/n): ").strip().lower() == 'y'
    
    if not output_dir:
        output_dir = os.getcwd()
    
    # Check if destination already contains valid subject folders
    required_folders = ['anchor', 'positive', 'negative']
    existing_subjects = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and
           all(os.path.isdir(os.path.join(output_dir, d, sub)) for sub in required_folders)
    ]

    if existing_subjects:
        logger.info(f"{len(existing_subjects)} subject folders already exist in destination. Skipping data preparation step.")
        processed_folders = [os.path.join(output_dir, d) for d in existing_subjects]
    else:
        # Initialize and run data preparation
        try:
            data_prep = DataPreparation(expert_properties_path, output_dir, skip_augmentation)
            processed_folders, skipped_folders = data_prep.prepare_data()
            
            # Final summary
            logger.info("\nProcessing complete!")
            logger.info(f"Successfully processed: {len(processed_folders)} folders")
            logger.info(f"Skipped: {len(skipped_folders)} folders")
            logger.info(f"All processed images are saved in: {output_dir}")
            
        except Exception as e:
            logger.error(f"Error during data preparation: {str(e)}")
            raise

    # --- Splitting logic ---
    DATA_DIR = output_dir
    OUTPUT_PREFIX = 'train_batch_'
    OUTPUT_DIR = os.path.join('dataset', 'training_splits')  # Save batch files in dataset/training_splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for existing .txt split files
    existing_txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    if existing_txt_files:
        user_choice = input(f"Existing split .txt files found in {OUTPUT_DIR}. Do you want to delete them and create new splits? (y/n): ").strip().lower()
        if user_choice == 'y':
            for f in existing_txt_files:
                os.remove(os.path.join(OUTPUT_DIR, f))
            print(f"Deleted {len(existing_txt_files)} existing split files.")
        else:
            print("Skipping splitting step. No new split files created.")
            return

    # 1. List all subject folders in processed data
    subject_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

    # Ask user for number of folders per training
    while True:
        try:
            user_input = input('Enter the number of folders per training (default: 100): ').strip()
            if user_input == '':
                num_folders_per_training = 100
            else:
                num_folders_per_training = int(user_input)
            if num_folders_per_training <= 0:
                print('Please enter a positive integer.')
                continue
            if num_folders_per_training > len(subject_folders):
                print(f'Cannot select more than {len(subject_folders)} folders.')
                continue
            break
        except ValueError:
            print('Please enter a valid integer.')

    # 2. Shuffle randomly
    random.shuffle(subject_folders)

    # 3. Divide into batches of num_folders_per_training
    batches = [subject_folders[i:i+num_folders_per_training] for i in range(0, len(subject_folders), num_folders_per_training)]

    # 4. Save each batch to a separate file
    for idx, batch in enumerate(batches, 1):
        batch_file = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}{idx}.txt')
        with open(batch_file, 'w') as f:
            for folder in batch:
                f.write(folder + '\n')
        print(f'Saved {len(batch)} folders to {batch_file}')

if __name__ == "__main__":
    main() 