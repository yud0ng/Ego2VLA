#!/usr/bin/env python3
"""
SmolVLA Experiment Runner

This script runs SmolVLA policy evaluation experiments in MuJoCo simulation.
You can configure model type, task, environment, timeout settings, and more.

Usage:
    python run_smolvla_experiment.py --task "pick and place" --pretrain --episodes 20
    python run_smolvla_experiment.py --model ./smolvla_model --timeout 120
    python run_smolvla_experiment.py --help
"""

import argparse
import time
import sys
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Add parent directory to Python path to find mujoco_env module
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features

# Local imports
from mujoco_env.y_env2 import SimpleEnv2


def get_default_transform(image_size: int = 224):
    """
    Returns a torchvision transform that converts PIL images to tensors.
    
    Args:
        image_size: Target image size (not used, kept for compatibility)
    
    Returns:
        transforms.Compose: Transform pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),  # PIL [0-255] -> FloatTensor [0.0-1.0], shape C×H×W
    ])


def setup_device():
    """
    Select the best available device (CUDA > MPS > CPU).
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        print("Device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Device: MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("Device: CPU (Warning: This will be slow)")
    
    return device


def load_dataset_metadata(dataset_name: str, dataset_root: str):
    """
    Load dataset metadata for policy configuration.
    
    Args:
        dataset_name: Name of the dataset
        dataset_root: Root directory of the dataset
    
    Returns:
        LeRobotDatasetMetadata: Dataset metadata object
    """
    print(f"\nLoading dataset metadata...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Root: {dataset_root}")
    
    try:
        dataset_metadata = LeRobotDatasetMetadata(dataset_name, root=dataset_root)
        print(f"  Status: Successfully loaded")
        return dataset_metadata
    except Exception as e:
        print(f"  Error: {e}")
        raise


def configure_policy(dataset_metadata, chunk_size: int = 5, n_action_steps: int = 5):
    """
    Configure SmolVLA policy from dataset metadata.
    
    Args:
        dataset_metadata: Dataset metadata object
        chunk_size: Number of action steps to predict at once
        n_action_steps: Number of action steps to execute
    
    Returns:
        tuple: (config, delta_timestamps)
    """
    print(f"\nConfiguring policy...")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Action steps: {n_action_steps}")
    
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps
    )
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    
    print(f"  Input features: {list(input_features.keys())}")
    print(f"  Output features: {list(output_features.keys())}")
    
    return cfg, delta_timestamps


def load_policy(model_path: str, cfg, dataset_metadata, device: str, use_pretrain: bool = False):
    """
    Load SmolVLA policy from checkpoint or HuggingFace.
    
    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        cfg: Policy configuration
        dataset_metadata: Dataset metadata
        device: Device to load model on
        use_pretrain: Whether to use pretrained model from HuggingFace
    
    Returns:
        SmolVLAPolicy: Loaded policy
    """
    print(f"\nLoading policy...")
    
    if use_pretrain:
        print(f"  Type: Pretrained (from HuggingFace)")
        print(f"  Model ID: {model_path}")
        print(f"  Note: Pretrained model may have low success rate (~5-10%)")
        
        try:
            policy = SmolVLAPolicy.from_pretrained(
                model_path,
                config=cfg,
                dataset_stats=dataset_metadata.stats
            )
            print(f"  Status: Successfully downloaded")
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Tip: Check network connection or use local model")
            raise
    else:
        print(f"  Type: Local finetuned model")
        print(f"  Path: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        policy = SmolVLAPolicy.from_pretrained(
            model_path,
            config=cfg,
            dataset_stats=dataset_metadata.stats,
            local_files_only=True
        )
        print(f"  Status: Successfully loaded")
    
    policy.to(device)
    policy.eval()
    print(f"  Device: {device}")
    
    return policy


def initialize_environment(scene_xml: str, action_type: str = 'joint_angle'):
    """
    Initialize MuJoCo simulation environment.
    
    Args:
        scene_xml: Path to MuJoCo scene XML file
        action_type: Type of action space ('joint_angle' or 'end_effector')
    
    Returns:
        SimpleEnv2: Initialized environment
    """
    print(f"\nInitializing environment...")
    print(f"  Scene: {scene_xml}")
    print(f"  Action type: {action_type}")
    
    if not Path(scene_xml).exists():
        raise FileNotFoundError(f"Scene file not found at {scene_xml}")
    
    env = SimpleEnv2(scene_xml, action_type=action_type)
    print(f"  Status: Environment initialized")
    
    return env


def run_experiment(
    env,
    policy,
    device: str,
    task: str = None,
    timeout_seconds: int = 60,
    max_episodes: int = 10,
    control_hz: int = 20,
    verbose: bool = True
):
    """
    Run experiment with multiple episodes.
    
    Args:
        env: MuJoCo environment
        policy: Loaded policy
        device: Device name
        task: Optional custom task instruction
        timeout_seconds: Timeout per episode (seconds)
        max_episodes: Maximum number of episodes to run
        control_hz: Control frequency (Hz)
        verbose: Whether to print progress updates
    
    Returns:
        dict: Experiment statistics
    """
    print(f"\n{'='*60}")
    print(f"Starting Experiment")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Timeout: {timeout_seconds} seconds")
    print(f"  Max episodes: {max_episodes}")
    print(f"  Control frequency: {control_hz} Hz")
    if task:
        print(f"  Custom task: {task}")
    print(f"{'-'*60}")
    
    # Statistics tracking
    episode_num = 0
    success_count = 0
    timeout_count = 0
    total_steps = 0
    episode_times = []
    
    IMG_TRANSFORM = get_default_transform()
    
    while env.env.is_viewer_alive() and episode_num < max_episodes:
        # Start new episode
        episode_num += 1
        step = 0
        start_time = time.time()
        
        print(f"\nEpisode {episode_num}/{max_episodes}")
        
        # Set custom task if provided
        if task:
            env.set_instruction(given=task)
        else:
            env.set_instruction()  # Random task
        
        print(f"Task: {env.instruction}")
        
        env.reset(seed=episode_num)
        policy.reset()
        
        episode_success = False
        episode_timeout = False
        
        # Episode main loop
        while env.env.is_viewer_alive():
            env.step_env()
            
            if env.env.loop_every(HZ=control_hz):
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    episode_timeout = True
                    timeout_count += 1
                    print(f"Timeout! ({elapsed_time:.1f}s > {timeout_seconds}s)")
                    break
                
                # Check if task is completed
                success = env.check_success()
                if success:
                    episode_success = True
                    success_count += 1
                    episode_times.append(elapsed_time)
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
                with torch.inference_mode():
                    action = policy.select_action(data)
                action = action[0, :7].cpu().detach().numpy()
                _ = env.step(action)
                env.render()
                
                step += 1
                total_steps += 1
                
                # Show progress every 5 seconds
                if verbose and int(elapsed_time) % 5 == 0 and int(elapsed_time) > 0 and step % (control_hz * 5) == 0:
                    print(f"Progress: {elapsed_time:.0f}s / {timeout_seconds}s | Steps: {step}")
        
        # Episode end statistics
        if not episode_success and not episode_timeout:
            print(f"Episode {episode_num} interrupted")
        
        # Brief delay before next episode
        time.sleep(0.5)
    
    # Calculate statistics
    success_rate = (success_count / episode_num * 100) if episode_num > 0 else 0
    timeout_rate = (timeout_count / episode_num * 100) if episode_num > 0 else 0
    failure_count = episode_num - success_count - timeout_count
    avg_steps = total_steps / episode_num if episode_num > 0 else 0
    avg_success_time = sum(episode_times) / len(episode_times) if episode_times else 0
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"Experiment Statistics")
    print(f"{'='*60}")
    print(f"Total Episodes: {episode_num}")
    print(f"Successes: {success_count} ({success_rate:.1f}%)")
    print(f"Timeouts: {timeout_count} ({timeout_rate:.1f}%)")
    print(f"Failures: {failure_count}")
    print(f"Total Steps: {total_steps}")
    print(f"Average Steps/Episode: {avg_steps:.1f}")
    if episode_times:
        print(f"Average Success Time: {avg_success_time:.1f}s")
    print(f"{'='*60}")
    
    return {
        'total_episodes': episode_num,
        'successes': success_count,
        'success_rate': success_rate,
        'timeouts': timeout_count,
        'timeout_rate': timeout_rate,
        'failures': failure_count,
        'total_steps': total_steps,
        'avg_steps': avg_steps,
        'avg_success_time': avg_success_time,
    }


def main():
    """Main function to run SmolVLA experiments."""
    parser = argparse.ArgumentParser(
        description='Run SmolVLA policy evaluation experiments in MuJoCo simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use pretrained model with default settings
  python run_smolvla_experiment.py --pretrain
  
  # Use local finetuned model with custom settings
  python run_smolvla_experiment.py --model ./smolvla_model --episodes 20 --timeout 120
  
  # Specify custom task and environment
  python run_smolvla_experiment.py --task "Place the red mug on the plate" --env ./asset/custom_scene.xml
  
  # Quick test with 3 episodes
  python run_smolvla_experiment.py --episodes 3 --timeout 30
        """
    )
    
    # Model arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='./smolvla_model',
        help='Path to model checkpoint or HuggingFace model ID (default: ./smolvla_model)'
    )
    parser.add_argument(
        '--pretrain', '-p',
        action='store_true',
        help='Use pretrained model from HuggingFace (lerobot/smolvla_base)'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='omy_pnp_language',
        help='Dataset name (default: omy_pnp_language)'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='./omy_pnp_language',
        help='Dataset root directory (default: ./omy_pnp_language)'
    )
    
    # Environment arguments
    parser.add_argument(
        '--env', '-e',
        type=str,
        default='./asset/example_scene_y2.xml',
        help='Path to MuJoCo scene XML file (default: ./asset/example_scene_y2.xml)'
    )
    parser.add_argument(
        '--action-type',
        type=str,
        default='joint_angle',
        choices=['joint_angle', 'end_effector'],
        help='Action space type (default: joint_angle)'
    )
    
    # Task arguments
    parser.add_argument(
        '--task', '-t',
        type=str,
        default=None,
        help='Custom task instruction (e.g., "Place the red mug on the plate"). If not set, uses random tasks.'
    )
    
    # Experiment arguments
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout per episode in seconds (default: 60)'
    )
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=10,
        help='Maximum number of episodes to run (default: 10)'
    )
    parser.add_argument(
        '--hz',
        type=int,
        default=20,
        help='Control frequency in Hz (default: 20)'
    )
    
    # Policy configuration arguments
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=5,
        help='Number of action steps to predict (default: 5)'
    )
    parser.add_argument(
        '--n-action-steps',
        type=int,
        default=5,
        help='Number of action steps to execute (default: 5)'
    )
    
    # Other arguments
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress updates during episodes'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Override model path if using pretrain
    if args.pretrain:
        args.model = "lerobot/smolvla_base"
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    try:
        # Setup device
        device = args.device if args.device else setup_device()
        
        # Load dataset metadata
        dataset_metadata = load_dataset_metadata(args.dataset, args.dataset_root)
        
        # Configure policy
        cfg, delta_timestamps = configure_policy(
            dataset_metadata,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps
        )
        
        # Load policy
        policy = load_policy(
            args.model,
            cfg,
            dataset_metadata,
            device,
            use_pretrain=args.pretrain
        )
        
        # Initialize environment
        env = initialize_environment(args.env, args.action_type)
        
        # Run experiment
        stats = run_experiment(
            env,
            policy,
            device,
            task=args.task,
            timeout_seconds=args.timeout,
            max_episodes=args.episodes,
            control_hz=args.hz,
            verbose=not args.quiet
        )
        
        # Return success if success rate > 50%
        if stats['success_rate'] > 50:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
