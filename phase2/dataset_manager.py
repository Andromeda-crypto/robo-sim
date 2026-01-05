# dataset_manager.py - Phase 2: Dataset management and indexing
#
# Creates and maintains dataset_index.csv with episode metadata.
# Usage:
#   python dataset_manager.py --update  # Update index from episodes directory
#   python dataset_manager.py --stats   # Show dataset statistics

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import argparse
import glob
from phase2.data_recorder import load_episode
from phase2.task_evaluation import evaluate_episode


def update_dataset_index(episodes_dir="data/episodes", index_file="data/dataset_index.csv"):
    """
    Scan episodes directory and create/update dataset index.
    
    Args:
        episodes_dir: Directory containing episode .npz files
        index_file: Path to CSV index file
    """
    if not os.path.exists(episodes_dir):
        print(f"Episodes directory not found: {episodes_dir}")
        return
    
    # Find all episode files
    episode_files = glob.glob(os.path.join(episodes_dir, "*.npz"))
    episode_files.sort()
    
    if len(episode_files) == 0:
        print(f"No episode files found in {episodes_dir}")
        return
    
    print(f"Found {len(episode_files)} episode files")
    print("Evaluating episodes and building index...")
    
    # Load existing index if it exists
    existing_entries = {}
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_entries[row['filename']] = row
    
    # Process each episode
    index_entries = []
    for filepath in episode_files:
        filename = os.path.basename(filepath)
        
        # Check if we already have this entry (skip if file hasn't changed)
        if filename in existing_entries:
            entry = existing_entries[filename]
            file_mtime = os.path.getmtime(filepath)
            if entry.get('file_mtime') == str(file_mtime):
                # File unchanged, use existing entry
                index_entries.append(entry)
                continue
        
        # Load and evaluate episode
        try:
            episode_data = load_episode(filepath)
            if episode_data is None:
                print(f"  Warning: Failed to load {filename}")
                continue
            
            # Evaluate episode
            result = evaluate_episode(episode_data)
            
            # Extract metadata
            metadata = episode_data.get('metadata', {})
            
            # Determine success status
            is_success = result['success']
            # Also check filename for _success suffix
            if '_success' in filename:
                is_success = True
            
            # Create index entry
            entry = {
                'filename': filename,
                'episode_id': metadata.get('episode_id', filename.replace('.npz', '')),
                'timestamp': metadata.get('timestamp', ''),
                'duration': f"{metadata.get('duration', 0):.2f}",
                'num_timesteps': str(metadata.get('num_timesteps', 0)),
                'success': 'yes' if is_success else 'no',
                'lifted': 'yes' if result['lifted'] else 'no',
                'in_target_zone': 'yes' if result['in_target_zone'] else 'no',
                'max_height': f"{result['max_height']:.3f}",
                'final_position_x': f"{result['final_position'][0]:.3f}",
                'final_position_y': f"{result['final_position'][1]:.3f}",
                'final_position_z': f"{result['final_position'][2]:.3f}",
                'file_mtime': str(os.path.getmtime(filepath)),
                'notes': ''
            }
            
            index_entries.append(entry)
            print(f"  Processed: {filename} - {'SUCCESS' if is_success else 'FAIL'}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    # Write index file
    if len(index_entries) == 0:
        print("No valid episodes to index")
        return
    
    fieldnames = [
        'filename', 'episode_id', 'timestamp', 'duration', 'num_timesteps',
        'success', 'lifted', 'in_target_zone', 'max_height',
        'final_position_x', 'final_position_y', 'final_position_z',
        'file_mtime', 'notes'
    ]
    
    with open(index_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_entries)
    
    print(f"\nDataset index updated: {index_file}")
    print(f"Total episodes: {len(index_entries)}")


def show_dataset_stats(index_file="data/dataset_index.csv"):
    """Display dataset statistics."""
    if not os.path.exists(index_file):
        print(f"Index file not found: {index_file}")
        print("Run 'python dataset_manager.py --update' first")
        return
    
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        entries = list(reader)
    
    if len(entries) == 0:
        print("No entries in index")
        return
    
    # Calculate statistics
    total = len(entries)
    successful = sum(1 for e in entries if e['success'] == 'yes')
    failed = total - successful
    
    lifted_count = sum(1 for e in entries if e['lifted'] == 'yes')
    in_zone_count = sum(1 for e in entries if e['in_target_zone'] == 'yes')
    
    # Duration statistics
    durations = [float(e['duration']) for e in entries]
    avg_duration = sum(durations) / len(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    
    # Height statistics
    heights = [float(e['max_height']) for e in entries]
    avg_height = sum(heights) / len(heights) if heights else 0
    max_height = max(heights) if heights else 0
    
    # Print statistics
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total episodes: {total}")
    print(f"  Successful: {successful} ({100*successful/total:.1f}%)")
    print(f"  Failed: {failed} ({100*failed/total:.1f}%)")
    print()
    print("Success Criteria:")
    print(f"  Lifted: {lifted_count}/{total} ({100*lifted_count/total:.1f}%)")
    print(f"  In target zone: {in_zone_count}/{total} ({100*in_zone_count/total:.1f}%)")
    print()
    print("Duration:")
    print(f"  Average: {avg_duration:.2f}s")
    print(f"  Min: {min_duration:.2f}s")
    print(f"  Max: {max_duration:.2f}s")
    print()
    print("Height:")
    print(f"  Average max height: {avg_height:.3f}m")
    print(f"  Maximum height: {max_height:.3f}m")
    print("=" * 60)
    
    # Show recent episodes
    print("\nRecent episodes (last 10):")
    print("-" * 60)
    recent = entries[-10:]
    for entry in recent:
        status = "✓" if entry['success'] == 'yes' else "✗"
        print(f"{status} {entry['filename']:40s} | {entry['duration']:6s}s | {entry['success']}")


def list_episodes(index_file="data/dataset_index.csv", filter_success=None):
    """List episodes in the dataset."""
    if not os.path.exists(index_file):
        print(f"Index file not found: {index_file}")
        return
    
    with open(index_file, 'r') as f:
        reader = csv.DictReader(f)
        entries = list(reader)
    
    if filter_success == 'success':
        entries = [e for e in entries if e['success'] == 'yes']
    elif filter_success == 'fail':
        entries = [e for e in entries if e['success'] == 'no']
    
    print(f"\nEpisodes ({len(entries)} total):")
    print("-" * 80)
    print(f"{'Filename':<40s} | {'Duration':<8s} | {'Success':<7s} | {'Lifted':<6s} | {'In Zone':<7s}")
    print("-" * 80)
    
    for entry in entries:
        print(f"{entry['filename']:<40s} | {entry['duration']:>8s} | "
              f"{entry['success']:<7s} | {entry['lifted']:<6s} | {entry['in_target_zone']:<7s}")


def main():
    parser = argparse.ArgumentParser(description='Manage teleoperation dataset')
    parser.add_argument('--update', action='store_true', 
                       help='Update dataset index from episodes directory')
    parser.add_argument('--stats', action='store_true',
                       help='Show dataset statistics')
    parser.add_argument('--list', action='store_true',
                       help='List all episodes')
    parser.add_argument('--filter', type=str, choices=['success', 'fail'],
                       help='Filter episodes (only with --list)')
    parser.add_argument('--episodes-dir', type=str, default='data/episodes',
                       help='Episodes directory (default: data/episodes)')
    parser.add_argument('--index-file', type=str, default='data/dataset_index.csv',
                       help='Index file path (default: data/dataset_index.csv)')
    
    args = parser.parse_args()
    
    if args.update:
        update_dataset_index(args.episodes_dir, args.index_file)
    elif args.stats:
        show_dataset_stats(args.index_file)
    elif args.list:
        list_episodes(args.index_file, args.filter)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python dataset_manager.py --update")
        print("  python dataset_manager.py --stats")
        print("  python dataset_manager.py --list")
        print("  python dataset_manager.py --list --filter success")


if __name__ == "__main__":
    main()

