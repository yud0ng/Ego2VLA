#!/usr/bin/env python3
"""
SmolVLA Policy Deployment Script

This script deploys a trained SmolVLA policy in a MuJoCo simulation environment.
Supports both pretrained and finetuned models with configurable parameters.

Usage Examples:
    # Use pretrained model with default settings
    python run_smolvla.py --pretrain
    
    # Use finetuned model with custom timeout
    python run_smolvla.py --model ./smolvla_model --timeout 120 --episodes 10
    
    # Custom task and environment
    python run_smolvla.py --task "Place the red mug on the plate" --xml ./asset/custom_scene.xml
    
    # Full configuration
    python run_smolvla.py --model ./smolvla_model --dataset ./omy_pnp_language --xml ./asset/example_scene_y2.xml --timeout 60 --episodes 20 --hz 20 --seed 42
"""

import argparse
import time
import sys
from pathlib import Path

# Add parent directory to Python path to find mujoco_env module
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features

from mujoco_env.y_env2 import SimpleEnv2


def get_default_transform(image_size: int = 224):
    """
    Returns a torchvision transform that converts PIL images to tensors.
    Scales pixel values from [0,255] to [0.0,1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(),  # PIL [0-255] -> FloatTensor [0.0-1.0], shape C×H×W
    ])


def setup_device():
    """
    Automatically select the best available device.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Device: {device}")
    return device


def load_dataset_metadata(dataset_name: str, dataset_root: str):
    """
    Load dataset metadata from the specified root directory.
    """
    try:
        dataset_metadata = LeRobotDatasetMetadata(dataset_name, root=dataset_root)
        print(f"Dataset root: {dataset_metadata.root}")
        return dataset_metadata
    except Exception as e:
        print(f"Failed to load dataset metadata: {e}")
        raise


def create_policy_config(dataset_metadata, chunk_size: int = 5, n_action_steps: int = 5):
    """
    Create SmolVLA policy configuration from dataset metadata.
    """
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps
    )
    
    return cfg


def load_policy(model_path: str, cfg, dataset_stats, device: str, pretrain: bool = False):
    """
    Load SmolVLA policy from either pretrained or finetuned model.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        cfg: SmolVLA configuration
        dataset_stats: Dataset statistics for normalization
        device: Device to load model on
        pretrain: If True, load from HuggingFace; if False, load local model
    """
    print("\n" + "=" * 50)
    
    if pretrain:
        print("Loading pretrained model from HuggingFace...")
        print(f"Model: {model_path}")
        print("Note: Pretrained model not trained on specific task, success rate may be low (~5-10%)")
        print()
        
        try:
            policy = SmolVLAPolicy.from_pretrained(
                model_path,
                config=cfg,
                dataset_stats=dataset_stats
            )
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Model download failed: {e}")
            print("Tip: Please check network connection or use local model")
            raise
    else:
        print("Loading finetuned model from local path...")
        print(f"Model path: {model_path}")
        print()
        
        try:
            policy = SmolVLAPolicy.from_pretrained(
                model_path,
                config=cfg,
                dataset_stats=dataset_stats,
                local_files_only=True
            )
            print("Local finetuned model loaded successfully!")
        except Exception as e:
            print(f"Failed to load local model: {e}")
            raise
    
    policy.to(device)
    print(f"Model moved to device: {device}")
    
    print("\nModel Information:")
    if pretrain:
        print(f"   - Type: SmolVLA Base (Pretrained)")
        print(f"   - Training Data: Open X-Embodiment (Multiple robot tasks)")
        print(f"   - Expected Success Rate: ~5-10% (Not finetuned on specific task)")
    else:
        print(f"   - Type: SmolVLA (Finetuned)")
        print(f"   - Training Data: Task-specific demonstrations")
        print(f"   - Expected Success Rate: ~60-80% (Finetuned on specific task)")
    
    print(f"   - Action Space: 7D (6 joints + 1 gripper)")
    print("=" * 50 + "\n")
    
    return policy


def run_experiment(env, policy, device, task: str = None,
                   timeout: int = 60, max_episodes: int = 20, hz: int = 20, seed: int = 42):
    """
    Run the experiment with the given policy and environment.
    
    Args:
        env: MuJoCo environment instance
        policy: SmolVLA policy instance
        device: Device for inference
        task: Optional custom task instruction
        timeout: Timeout duration in seconds
        max_episodes: Maximum number of episodes to run
        hz: Control frequency in Hz
        seed: Random seed for reproducibility
    """
    # Statistics tracking
    episode_num = 0
    success_count = 0
    timeout_count = 0
    total_steps = 0
    
    print("Starting experiment...")
    print(f"Timeout setting: {timeout} seconds")
    print(f"Max episodes: {max_episodes}")
    print(f"Control frequency: {hz} Hz")
    if task:
        print(f"Custom task: {task}")
    print("-" * 50)
    
    IMG_TRANSFORM = get_default_transform()
    
    while env.env.is_viewer_alive() and episode_num < max_episodes:
        # Start new episode
        episode_num += 1
        step = 0
        start_time = time.time()
        
        print(f"\nEpisode {episode_num}/{max_episodes}")
        
        # Override task instruction if provided
        if task:
            env.instruction = task
        
        print(f"Task: {env.instruction}")
        
        env.reset(seed=seed + episode_num)
        policy.reset()
        policy.eval()
        
        episode_success = False
        episode_timeout = False
        
        # Episode main loop
        while env.env.is_viewer_alive():
            env.step_env()
            
            if env.env.loop_every(HZ=hz):
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    episode_timeout = True
                    timeout_count += 1
                    print(f"Timeout! ({elapsed_time:.1f}s > {timeout}s)")
                    break
                
                # Check if task is completed
                success = env.check_success()
                if success:
                    episode_success = True
                    success_count += 1
                    print(f'Success! Time: {elapsed_time:.1f}s, Steps: {step}')
                    break
                
                # Get current state
                state = env.get_joint_state()[:6]
                image, wrist_image = env.grab_image()
                
                # Preprocess images
                image = Image.fromarray(image).resize((256, 256))
                image = IMG_TRANSFORM(image)
                wrist_image = Image.fromarray(wrist_image).resize((256, 256))
                wrist_image = IMG_TRANSFORM(wrist_image)
                
                # Build input
                data = {
                    'observation.state': torch.tensor([state]).to(device),
                    'observation.image': image.unsqueeze(0).to(device),
                    'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
                    'task': [env.instruction],
                }
                
                # Inference and execute action
                with torch.no_grad():
                    action = policy.select_action(data)
                action = action[0, :7].cpu().detach().numpy()
                _ = env.step(action)
                env.render()
                
                step += 1
                total_steps += 1
                
                # Show progress every 5 seconds
                if int(elapsed_time) % 5 == 0 and int(elapsed_time) > 0 and step % (hz * 5) == 0:
                    print(f"Progress: {elapsed_time:.0f}s / {timeout}s | Steps: {step}")
        
        # Episode end statistics
        if not episode_success and not episode_timeout:
            print(f"Episode {episode_num} interrupted")
        
        # Brief delay before next episode
        time.sleep(0.5)
    
    # Final Statistics
    print("\n" + "=" * 50)
    print("Experiment Statistics")
    print("=" * 50)
    print(f"Total Episodes: {episode_num}")
    if episode_num > 0:
        print(f"Successes: {success_count} ({success_count/episode_num*100:.1f}%)")
        print(f"Timeouts: {timeout_count} ({timeout_count/episode_num*100:.1f}%)")
        print(f"Failures: {episode_num - success_count - timeout_count}")
        print(f"Total Steps: {total_steps}")
        print(f"Average Steps/Episode: {total_steps/episode_num:.1f}")
    print("=" * 50)
    
    return {
        'total_episodes': episode_num,
        'success_count': success_count,
        'timeout_count': timeout_count,
        'total_steps': total_steps
    }


def main():
    parser = argparse.ArgumentParser(
        description='Deploy SmolVLA policy in MuJoCo simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use pretrained model
  python run_smolvla.py --pretrain
  
  # Use finetuned model with custom settings
  python run_smolvla.py --model ./smolvla_model --timeout 120 --episodes 10
  
  # Custom task
  python run_smolvla.py --task "Place the red mug on the plate" --episodes 5
  
  # Full configuration
  python run_smolvla.py --model ./smolvla_model --dataset ./omy_pnp_language \\
      --xml ./asset/example_scene_y2.xml --timeout 60 --episodes 20 --hz 20
        """
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='./smolvla_model',
                        help='Path to model directory or HuggingFace model ID (default: ./smolvla_model)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Use pretrained model from HuggingFace (lerobot/smolvla_base)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='omy_pnp_language',
                        help='Dataset name (default: omy_pnp_language)')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Dataset root path (default: auto-detect from ./demo_data_language or ./omy_pnp_language)')
    
    # Environment arguments
    parser.add_argument('--xml', type=str, default='./asset/example_scene_y2.xml',
                        help='Path to MuJoCo scene XML file (default: ./asset/example_scene_y2.xml)')
    parser.add_argument('--action-type', type=str, default='joint_angle',
                        choices=['joint_angle', 'joint_velocity'],
                        help='Action type for environment (default: joint_angle)')
    
    # Task arguments
    parser.add_argument('--task', type=str, default=None,
                        help='Custom task instruction (default: use environment default)')
    
    # Experiment arguments
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout duration in seconds (default: 60)')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Maximum number of episodes (default: 20)')
    parser.add_argument('--hz', type=int, default=20,
                        help='Control frequency in Hz (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Policy configuration arguments
    parser.add_argument('--chunk-size', type=int, default=5,
                        help='Policy chunk size (default: 5)')
    parser.add_argument('--n-action-steps', type=int, default=5,
                        help='Number of action steps (default: 5)')
    
    args = parser.parse_args()
    
    # Override model path if using pretrain
    if args.pretrain:
        args.model = 'lerobot/smolvla_base'
    
    # Auto-detect and fix XML path (check parent directory if not found in current)
    xml_path = Path(args.xml)
    if not xml_path.exists() and not xml_path.is_absolute():
        # Try parent directory
        parent_xml = parent_dir / args.xml.lstrip('./')
        if parent_xml.exists():
            args.xml = str(parent_xml)
        else:
            # Try as relative to parent directory
            alt_path = parent_dir / 'asset' / Path(args.xml).name
            if alt_path.exists():
                args.xml = str(alt_path)
    
    # Auto-detect dataset root if not provided
    if args.dataset_root is None:
        # Check both current directory and parent directory
        possible_paths = [
            Path('./demo_data_language'),
            Path('./omy_pnp_language'),
            Path('../demo_data_language'),
            Path('../omy_pnp_language'),
        ]
        
        for path in possible_paths:
            if path.exists():
                args.dataset_root = str(path)
                break
        
        if args.dataset_root is None:
            print("Error: Could not find dataset directory")
            print("Please specify --dataset-root or ensure one of these exists:")
            print("  ./demo_data_language or ./omy_pnp_language")
            print("  ../demo_data_language or ../omy_pnp_language")
            sys.exit(1)
    
    print("=" * 50)
    print("SmolVLA Policy Deployment")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Pretrain: {args.pretrain}")
    print(f"Dataset: {args.dataset} (root: {args.dataset_root})")
    print(f"Scene XML: {args.xml}")
    print(f"Timeout: {args.timeout}s")
    print(f"Episodes: {args.episodes}")
    print(f"Frequency: {args.hz} Hz")
    if args.task:
        print(f"Task: {args.task}")
    print("=" * 50 + "\n")
    
    try:
        # Setup device
        device = setup_device()
        
        # Load dataset metadata
        print("\nLoading dataset metadata...")
        dataset_metadata = load_dataset_metadata(args.dataset, args.dataset_root)
        
        # Create policy configuration
        print("Creating policy configuration...")
        cfg = create_policy_config(
            dataset_metadata,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps
        )
        delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
        
        # Load policy
        policy = load_policy(
            args.model,
            cfg,
            dataset_metadata.stats,
            device,
            pretrain=args.pretrain
        )
        
        # Create environment
        print("Creating MuJoCo environment...")
        env = SimpleEnv2(args.xml, action_type=args.action_type)
        print(f"Environment created: {args.xml}")
        print(f"Default task: {env.instruction}\n")
        
        # Run experiment
        results = run_experiment(
            env=env,
            policy=policy,
            device=device,
            task=args.task,
            timeout=args.timeout,
            max_episodes=args.episodes,
            hz=args.hz,
            seed=args.seed
        )
        
        print("\nExperiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
