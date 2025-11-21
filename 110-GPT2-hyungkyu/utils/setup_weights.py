"""
Download and convert GPT-2 weights in one step
Uses existing functions from download_gpt2_weights.py and convert_openai_weights.py
"""

import sys
import os
import argparse

def main():
    """Main function to download and convert GPT-2 weights"""

    parser = argparse.ArgumentParser(description='Download and convert GPT-2 weights')
    parser.add_argument('--model', type=str, default='124M',
                       choices=['124M', '355M', '774M', '1558M'],
                       help='GPT-2 model size (default: 124M)')
    parser.add_argument('--output-dir', type=str, default='assets/weights',
                       help='Output directory (default: assets/weights)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download and re-conversion even if files exist')

    args = parser.parse_args()

    checkpoint_dir = f"{args.output_dir}/{args.model}"
    weights_file = f"{checkpoint_dir}/gpt2_weights.bin"
    config_file = f"{checkpoint_dir}/gpt2_config.txt"

    print("="*60)
    print(f"GPT-2 Weights Setup - Model: {args.model}")
    print("="*60)

    # Check if converted weights already exist
    if not args.force and os.path.exists(weights_file) and os.path.exists(config_file):
        print(f"\n✓ Weights already exist:")
        print(f"  - {weights_file}")
        print(f"  - {config_file}")
        print(f"\nSetup already complete! No download or conversion needed.")
        print(f"\nTo force re-download/conversion, use --force flag:")
        print(f"  python utils/setup_weights.py --model {args.model} --force")
        return 0

    # Step 1: Download checkpoint from OpenAI using existing download function
    print("\n[Step 1/2] Downloading OpenAI checkpoint...")
    print("(Files that already exist will be skipped)")
    try:
        from download_gpt2_weights import download_and_load_gpt2

        download_and_load_gpt2(model_size=args.model, models_dir=args.output_dir)
        print("\n✓ Download complete!")

    except ImportError as e:
        print(f"\n✗ Failed to import download_gpt2_weights: {e}")
        print("Make sure download_gpt2_weights.py is in the same directory.")
        return 1
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("Check your internet connection and try again.")
        return 1

    # Step 2: Convert to binary using existing convert function
    print("\n[Step 2/2] Converting to binary format...")

    # Check if conversion is needed
    if not args.force and os.path.exists(weights_file) and os.path.exists(config_file):
        print(f"✓ Converted weights already exist, skipping conversion")
    else:
        try:
            from convert_openai_weights import convert_checkpoint

            convert_checkpoint(checkpoint_dir, checkpoint_dir)
        except ImportError as e:
            print(f"\n✗ Failed to import convert_openai_weights: {e}")
            print("Make sure convert_openai_weights.py is in the same directory.")
            return 1
        except Exception as e:
            print(f"\n✗ Conversion failed: {e}")
            print("\nMake sure you have the required packages installed:")
            print("  pip install numpy tensorflow")
            return 1

    # Final success message
    print("\n" + "="*60)
    print("✓ Setup complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {checkpoint_dir}/gpt2_weights.bin")
    print(f"  - {checkpoint_dir}/gpt2_config.txt")
    print(f"\nYou can now build and run the project.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
