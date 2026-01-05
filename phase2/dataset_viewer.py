# dataset_viewer.py - Phase 2: Interactive dataset browser
#
# Browse and visualize episodes in the dataset.
# Usage:
#   python dataset_viewer.py

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import argparse
from phase2.data_recorder import load_episode
from phase2.replay import replay_single_episode


def browse_dataset(index_file="data/dataset_index.csv", episodes_dir="data/episodes"):
    """Interactive dataset browser."""
    if not os.path.exists(index_file):
        print(f"Index file not found: {index_file}")
        print("Run 'python dataset_manager.py --update' first")
        return
    
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        entries = list(reader)
    
    if len(entries) == 0:
        print("No episodes in dataset")
        return
    
    print("=" * 80)
    print("DATASET BROWSER")
    print("=" * 80)
    print(f"Total episodes: {len(entries)}")
    print()
    
    # Show summary
    successful = [e for e in entries if e['success'] == 'yes']
    failed = [e for e in entries if e['success'] == 'no']
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()
    
    # List episodes
    print("Episodes:")
    print("-" * 80)
    for i, entry in enumerate(entries):
        status = "✓" if entry['success'] == 'yes' else "✗"
        print(f"{i+1:3d}. {status} {entry['filename']:<40s} | "
              f"{entry['duration']:>6s}s | {entry['success']}")
    
    print()
    print("Commands:")
    print("  Enter episode number to replay")
    print("  's' to show only successful episodes")
    print("  'f' to show only failed episodes")
    print("  'a' to show all episodes")
    print("  'q' to quit")
    print()
    
    # Interactive loop
    show_all = True
    filtered_entries = entries
    
    while True:
        try:
            command = input("> ").strip().lower()
            
            if command == 'q':
                break
            elif command == 's':
                filtered_entries = successful
                show_all = False
                print(f"\nShowing {len(filtered_entries)} successful episodes:")
                for i, entry in enumerate(filtered_entries):
                    print(f"{i+1:3d}. {entry['filename']}")
            elif command == 'f':
                filtered_entries = failed
                show_all = False
                print(f"\nShowing {len(filtered_entries)} failed episodes:")
                for i, entry in enumerate(filtered_entries):
                    print(f"{i+1:3d}. {entry['filename']}")
            elif command == 'a':
                filtered_entries = entries
                show_all = True
                print(f"\nShowing all {len(filtered_entries)} episodes")
            elif command.isdigit():
                episode_num = int(command) - 1
                if 0 <= episode_num < len(filtered_entries):
                    entry = filtered_entries[episode_num]
                    filepath = os.path.join(episodes_dir, entry['filename'])
                    
                    if os.path.exists(filepath):
                        print(f"\nReplaying: {entry['filename']}")
                        print("Close PyBullet window after replay to return to browser")
                        replay_single_episode(filepath, playback_speed=1.0, show_gui=True)
                        print("\nReturned to browser")
                    else:
                        print(f"File not found: {filepath}")
                else:
                    print(f"Invalid episode number. Enter 1-{len(filtered_entries)}")
            elif command == '':
                continue
            else:
                print("Unknown command. Use 'q' to quit, number to replay, 's'/'f'/'a' to filter")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Browse and view dataset episodes')
    parser.add_argument('--index-file', type=str, default='data/dataset_index.csv',
                       help='Index file path (default: data/dataset_index.csv)')
    parser.add_argument('--episodes-dir', type=str, default='data/episodes',
                       help='Episodes directory (default: data/episodes)')
    
    args = parser.parse_args()
    browse_dataset(args.index_file, args.episodes_dir)


if __name__ == "__main__":
    main()

